import os
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel, LoraConfig
from pathlib import Path
from anticipation.sample import generate
from anticipation.convert import events_to_midi, midi_to_events

# Configuration
MODEL_NAME = 'stanford-crfm/music-large-800k'
CHECKPOINT_PATH = os.path.join('training_output', 'checkpoint-best')
DEVICE = torch.device("cuda")

base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Load the fine-tuned adapter checkpoint and wrap the base model with PEFT
model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
model.to(DEVICE)
model.eval()

input_dir = Path("test_input")
output_dir = Path("output_midi")
output_dir.mkdir(parents=True, exist_ok=True)

for input_path in input_dir.glob("*.mid"):
    print(f"Processing {input_path} ...")
    # Load the tokenized input tensor and add a batch dimension if necessary
    input_events = midi_to_events(str(input_path), debug=False)
    input_tensor = torch.tensor(input_events).unsqueeze(0).to(DEVICE)

    # Truncate input if it exceeds available context length, leaving room for new tokens.
    max_new_tokens = 128  # adjust as needed
    max_input_length = model.config.n_positions - max_new_tokens
    if input_tensor.size(1) > max_input_length:
        input_tensor = input_tensor[:, -max_input_length:]
    
    # Create an attention mask for the input tensor
    attention_mask = torch.ones_like(input_tensor)
    
    with torch.no_grad():
        generated_tokens = model.generate(
            input_tensor, 
            max_new_tokens=max_new_tokens,
            attention_mask=attention_mask
        )
    
    try:
        token_list = generated_tokens[0].cpu().tolist()
        midi_object = events_to_midi(token_list)
        # Create an output file name: e.g., "your_input_file_aligned.mid"
        midi_file_path = output_dir / f"{input_path.stem}_aligned.mid"
        midi_object.save(str(midi_file_path))
        print(f"MIDI file saved to {midi_file_path}")
    except Exception as e:
        print(f"Error converting tokens to MIDI for {input_path}: {e}")