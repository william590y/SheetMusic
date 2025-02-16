"""
Fine-tuning script for the Anticipatory Music Transformer using paired MIDI files.
This script defines configuration parameters internally.
"""

import os
import torch
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from anticipation.convert import midi_to_events, events_to_midi
from anticipation.tokenize import tokenize, detokenize


# ----------------------
# Configuration Settings
# ----------------------

MODEL_NAME = 'stanford-crfm/music-medium-800k'
INPUT_DIR = 'dataset/input'
TARGET_DIR = 'dataset/target'
OUTPUT_DIR = 'training_output'

EPOCHS = 10
BATCH_SIZE = 4
LEARNING_RATE = 5e-5
GRAD_ACCUM = 8
LOG_INTERVAL = 10
SAVE_INTERVAL = 1

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# ----------------------
# Dataset and Collation
# ----------------------

class MIDIPairDataset(Dataset):
    def __init__(self, input_dir, target_dir):
        self.pairs = []

        input_files = set(os.listdir(input_dir))
        target_files = set(os.listdir(target_dir))
        common_files = input_files & target_files

        if not common_files:
            raise ValueError("No common filenames found between input and output directories")
        
        for fname in common_files:
            input_path = os.path.join(input_dir, fname)
            target_path = os.path.join(target_dir, fname)

            input_events = midi_to_events(input_path)
            target_events = midi_to_events(target_path)

            input_tokens = tokenize(input_events)
            target_tokens = tokenize(target_events)

            input_tensor = torch.tensor(input_tokens, dtype=torch.long)
            target_tensor = torch.tensor(target_tokens, dtype=torch.long)

            self.pairs.append((input_tensor, target_tensor))

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        return self.pairs[idx]
    
def collate_fn(batch):
    inputs, targets = zip(*batch)

    padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    padded_targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=-100)

    return padded_inputs, padded_targets

# ----------------------
# Training Loop
# ----------------------

def main():
    # Initialize the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # Use GPU
    device = torch.device("cuda")
    model.to(device)

    # Create dataset and dataloader
    dataset = MIDIPairDataset(INPUT_DIR, TARGET_DIR)
    loader = DataLoader(dataset,  batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # Prepare optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training Loop
    global_step = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batchidx, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Attention mask based on non-zero tokens
            attention_mask = (inputs != 0).float()

            outputs = model(inputs, labels=targets, attention_mask=attention_mask)
            loss = outputs.loss/GRAD_ACCUM

            loss.backward()

            if (batchidx+1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                global_step += 1
                total_loss += loss.item()

                if global_step % LOG_INTERVAL == 0:
                    avg_loss = total_loss / LOG_INTERVAL
                    print(f"Epoch {epoch + 1} | Step {global_step} | Loss {avg_loss:.4f}")
                    total_loss = 0

        if (epoch+1) % SAVE_INTERVAL == 0:
            ckpt_path = os.path.join(OUTPUT_DIR, f"checkpoint-{epoch+1}")
            Path(ckpt_path).mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

if __name__ == "__main__":
    # Ensure output directory exists
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    main()