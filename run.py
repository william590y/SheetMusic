import os
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel, LoraConfig
from pathlib import Path
from anticipation.sample import generate
from anticipation.convert import events_to_midi, midi_to_events
from anticipation import ops
import traceback
import torch.nn.functional as F
from anticipation.config import MAX_DUR


def unpad(padded_tensor, pad_value=-100):
    # Find the index of the first pad token (if any) and slice to that length.
    non_pad_indices = (padded_tensor != pad_value).nonzero(as_tuple=True)[0]
    if non_pad_indices.numel() > 0:
        last_index = non_pad_indices[-1].item() + 1
    else:
        last_index = padded_tensor.size(0)
    return padded_tensor[:last_index]

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

input_dir = Path(r"training_output/test_input")
output_dir = Path("output_midi")
output_dir.mkdir(parents=True, exist_ok=True)

max_len = 942

for input_path in input_dir.glob("*.mid"):
    print(f"Processing {input_path} ...")
    # Load the tokenized input tensor and add a batch dimension if necessary
    input_events = midi_to_events(str(input_path), debug=False)
    input_tensor = torch.tensor(input_events).unsqueeze(0).to(DEVICE)
    input_tensor = F.pad(input_tensor, (0, max_len - input_tensor.size(1)), value=0)
    
    # Create an attention mask for the input tensor
    attention_mask = torch.ones_like(input_tensor)
    
    with torch.no_grad():
        generated_tokens = model.generate(
            input_tensor, 
            attention_mask=attention_mask
        )
    
    try:
        token_list = generated_tokens[0].cpu().tolist()
        token_list = [max(0, tok) for tok in token_list]
        for i in range(1, len(token_list), 5):
            if token_list[i] >= MAX_DUR:
                token_list[i] = MAX_DUR - 1
        token_tensor = torch.tensor(token_list)
        token_list = unpad(token_tensor, pad_value= 50256)
        midi_object = events_to_midi(token_list)
        # Create an output file name: e.g., "your_input_file_aligned.mid"
        midi_file_path = output_dir / f"{input_path.stem}_aligned.mid"
        midi_object.save(str(midi_file_path))
        print(f"MIDI file saved to {midi_file_path}")
    except Exception as e:
        print(traceback.format_exc())
        print(f"Error converting tokens to MIDI for {input_path}: {e}")