"""
Fine-tuning script for the Anticipatory Music Transformer using paired MIDI files.
This script defines configuration parameters internally.
"""

import os
import csv
import torch
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM, AdamW
from anticipation.convert import midi_to_events, events_to_midi
from anticipation.tokenize import tokenize, maybe_tokenize
from anticipation.vocab import VOCAB_SIZE
from anticipation.convert import midi_to_compound, compound_to_events
from anticipation import ops
import concurrent.futures
from tqdm import tqdm
from peft import get_peft_model, LoraConfig


# ----------------------
# Configuration Settings
# ----------------------

MODEL_NAME = 'stanford-crfm/music-large-800k'
INPUT_DIR = 'dataset/input'
TARGET_DIR = 'dataset/target'
OUTPUT_DIR = 'training_output'

EPOCHS = 10
BATCH_SIZE = 2 # changed from 4
LEARNING_RATE = 5e-5
GRAD_ACCUM = 2 # changed from 4
LOG_INTERVAL = 1
SAVE_INTERVAL = 1

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# ----------------------
# Dataset and Collation
# ----------------------

def process_file(fname, input_dir, target_dir):
    input_path = os.path.join(input_dir, fname)
    target_path = os.path.join(target_dir, fname)

    # MIDI to compound events (5-token groups)
    input_compound = midi_to_compound(input_path, debug=True)
    target_compound = midi_to_compound(target_path, debug=True)

    # Convert compound tokens to event tokens (3-token groups)
    input_events = compound_to_events(input_compound)
    target_events = compound_to_events(target_compound)

    # Validate length of event token sequence (should be multiple of 3)
    if len(input_events) % 3 != 0 or len(target_events) % 3 != 0:
        raise ValueError(f"Invalid compound sequence length in {fname}")

    print(f"Processed {fname} with {len(input_events)//3} input tokens and {len(target_events)//3} target tokens")
    return torch.tensor(input_events), torch.tensor(target_events)

class MIDIPairDataset(Dataset):
    def __init__(self, input_dir, target_dir):
        self.pairs = []

        input_files = set(os.listdir(input_dir))
        target_files = set(os.listdir(target_dir))
        common_files = input_files & target_files

        if not common_files:
            raise ValueError("No common filenames found between input and output directories")
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_file, fname, input_dir, target_dir) for fname in common_files]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing MIDI files"):
                self.pairs.append(future.result())

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        return self.pairs[idx]
    
def collate_fn(batch):
    inputs, targets = zip(*batch)

    max_len = max(
        max(inp.size(0) for inp in inputs),
        max(tgt.size(0) for tgt in targets)
    )

    # Pad both inputs and targets to the same max_len
    padded_inputs = torch.stack([
        torch.nn.functional.pad(inp, (0, max_len - inp.size(0)), value=0)
        for inp in inputs
    ])
    padded_targets = torch.stack([
        torch.nn.functional.pad(tgt, (0, max_len - tgt.size(0)), value=-100)
        for tgt in targets
    ])

    return padded_inputs, padded_targets

# ----------------------
# Training Loop
# ----------------------

def main():
    # Initialize training logs
    training_loss_log = []
    validation_loss_log = []

    # Initialize the model (rip no tokenizer from HF)
    #tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).cuda()

    lora_config = LoraConfig(
    r=8,                    # Low rank dimension (tweak as needed)
    lora_alpha=16,          # Scaling factor for LoRA (tweak as needed)
    target_modules=["c_attn"],  # List of target modules. Adjust based on model.
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
    
    model = get_peft_model(model, lora_config)
    print("Model is now wrapped with PEFT adapters.")

    # Use GPU
    device = torch.device("cuda")
    model.to(device)

    # Create dataset and dataloader
    dataset = MIDIPairDataset(INPUT_DIR, TARGET_DIR)

    # Split dataset into train, validation, and test splits
    total_size = len(dataset)
    train_size = int(TRAIN_SPLIT * total_size)
    val_size = int(VAL_SPLIT * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator = generator)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                            shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                          shuffle=False, collate_fn=collate_fn)

    # Dump tokenized test dataset to disk without detokenizing
    test_input_dir = os.path.join(OUTPUT_DIR, "test_input")
    test_target_dir = os.path.join(OUTPUT_DIR, "test_target")
    Path(test_input_dir).mkdir(parents=True, exist_ok=True)
    Path(test_target_dir).mkdir(parents=True, exist_ok=True)

    for idx in test_dataset.indices:
        input_tensor, target_tensor = dataset[idx]
        input_file = os.path.join(test_input_dir, f"test_input_{idx}.mid")
        target_file = os.path.join(test_target_dir, f"test_target_{idx}.mid")
        torch.save(events_to_midi(input_tensor.int().tolist()), input_file)
        torch.save(events_to_midi(target_tensor.int().tolist()), target_file)

    print(f"Test dataset files saved to {test_input_dir} and {test_target_dir}")

    # Prepare optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Determine maximum allowed sequence length from the model configuration
    max_length = model.config.n_positions

    # Training Loop
    best_val_loss = float('inf')
    global_step = 0
    total_loss = 0

    total_batches = EPOCHS * len(train_loader)
    pbar = tqdm(total=total_batches, desc="Training Progress", unit="batch")

    for epoch in range(EPOCHS):
        model.train()
        train_steps = 0

        for batchidx, (inputs, targets) in enumerate(train_loader):
            # Truncate inputs and targets if they exceed max_length
            if inputs.size(1) > max_length:
                inputs = inputs[:, :max_length]
                targets = targets[:, :max_length]

            inputs = inputs.to(device).long()
            targets = targets.to(device)

            # Attention mask based on non-zero tokens
            attention_mask = (inputs != 0).float()

            outputs = model(inputs, labels=targets, attention_mask=attention_mask)
            loss = outputs.loss
            loss.backward()

            if (batchidx+1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1
                total_loss += loss.item()
                train_steps += 1

                if global_step % LOG_INTERVAL == 0:
                    avg_loss = total_loss / train_steps
                    # print(f"Epoch {epoch + 1} | Step {global_step} | Loss {avg_loss:.4f}")
                    # Log the training loss
                    training_loss_log.append((epoch + 1, global_step, avg_loss))
                    total_loss = 0
                    train_steps = 0

            pbar.update(1)
            pbar.set_postfix({'Epoch': epoch+1, 'Loss': f"{loss.item():.4f}"})

        # Validation phase
        model.eval()
        total_val_loss = 0
        val_steps = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                if inputs.size(1) > max_length:
                    inputs = inputs[:, :max_length]
                    targets = targets[:, :max_length]
                inputs = inputs.to(device)
                targets = targets.to(device)
                attention_mask = (inputs != 0).float()

                outputs = model(inputs, labels=targets, attention_mask=attention_mask)
                total_val_loss += outputs.loss.item()
                val_steps += 1

        avg_val_loss = total_val_loss / val_steps
        print(f"Epoch {epoch + 1} | Validation Loss: {avg_val_loss:.4f}")

        # Log the validation loss
        validation_loss_log.append((epoch + 1, avg_val_loss))

        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(OUTPUT_DIR, "checkpoint-best")
            Path(ckpt_path).mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_path)
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")

        if (epoch+1) % SAVE_INTERVAL == 0:
            ckpt_path = os.path.join(OUTPUT_DIR, f"checkpoint-{epoch+1}")
            Path(ckpt_path).mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

        scheduler.step()

    # Write the losses to a csv file
    csv_path = os.path.join(OUTPUT_DIR, "loss_logs.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Epoch", "Global_Step", "Train_Loss", "Validation_Loss"])
        max_steps = max([item[1] for item in training_loss_log], default=0)
        for epoch, val_loss in validation_loss_log:
            # Use most recent training loss for that epoch
            train_loss = next((tl for (e, step, tl) in training_loss_log if e == epoch), None)
            writer.writerow([epoch, max_steps, train_loss if train_loss is not None else "", val_loss])

    print(f"Training complete. Loss logs saved to {csv_path}")

if __name__ == "__main__":
    # Ensure output directory exists
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    main()