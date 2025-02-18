import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig
from pathlib import Path

# Configuration
MODEL_NAME = 'stanford-crfm/music-large-800k'
CHECKPOINT_PATH = os.path.join('training_output', 'checkpoint-best')
DEVICE = torch.device("cuda")

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

# Set up the LoRA configuration (must match training)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Wrap the model with PEFT adapters and load the fine-tuned adapter checkpoint
model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
model.to(DEVICE)
model.eval()

# (Optional) Load a tokenizer if you need to map tokens back to text or MIDI events.
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load an arbitrary input file. This example assumes the file contains a tensor saved with torch.save().
input_file = os.path.join('path_to_input_files', 'your_input_file.pt')
if not Path(input_file).exists():
    raise FileNotFoundError(f"Input file not found: {input_file}")
input_tensor = torch.load(input_file).unsqueeze(0).to(DEVICE)  # unsqueeze to add batch dim if necessary

# Run inference. Adjust generation parameters as needed.
with torch.no_grad():
    generated_output = model.generate(input_tensor, max_length=model.config.n_positions)

print("Generated output tokens:")
print(generated_output.cpu().numpy())

# (Optional) If you have a detokenization function, call it here.
# midi_output = detokenize(generated_output)
# midi_output.save("output.mid")