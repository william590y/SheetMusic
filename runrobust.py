import os
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
from pathlib import Path
from anticipation.convert import events_to_midi, midi_to_events
import traceback
from tqdm import tqdm

def process_midi_file(input_path, output_path, model=None):
    """Process a MIDI file, attempting generation if model provided."""
    try:
        # Step 1: Load the input events
        input_events = midi_to_events(str(input_path))
        
        # Ensure we have groups of 3
        if len(input_events) % 3 != 0:
            input_events = input_events[:(len(input_events) // 3) * 3]
        
        # Step 2: Try model generation if provided
        if model is not None:
            try:
                # Prepare input
                input_tensor = torch.tensor(input_events).unsqueeze(0).to(model.device)
                
                # Create attention mask
                attention_mask = torch.ones_like(input_tensor)
                
                # Generate with model
                with torch.no_grad():
                    outputs = model.generate(
                        input_tensor,
                        attention_mask=attention_mask,
                        max_length=min(len(input_events) + 300, 2048),
                        pad_token_id=50256,
                        do_sample=True,
                        temperature=0.8
                    )
                
                # Get generated events
                generated_events = outputs[0].cpu().tolist()
                
                # Ensure we have groups of 3
                if len(generated_events) % 3 != 0:
                    generated_events = generated_events[:(len(generated_events) // 3) * 3]
                
                # Convert to MIDI - try with generated events first
                try:
                    midi_object = events_to_midi(generated_events)
                    midi_object.save(str(output_path))
                    print(f"✅ Generation successful: {output_path.name}")
                    return True
                except Exception:
                    # Generation failed, fall back to direct conversion
                    print(f"⚠️ Generation failed for {input_path.name}, falling back to direct conversion")
                    midi_object = events_to_midi(input_events)
                    midi_object.save(str(output_path))
                    print(f"✅ Direct conversion successful: {output_path.name}")
                    return True
                
            except Exception as e:
                print(f"⚠️ Model error: {e}")
                # Fall back to direct conversion
        
        # Step 3: Direct conversion (if no model or model failed)
        midi_object = events_to_midi(input_events)
        midi_object.save(str(output_path))
        print(f"✅ Direct conversion successful: {output_path.name}")
        return True
        
    except Exception as e:
        print(f"❌ Error processing {input_path.name}:")
        print(traceback.format_exc())
        return False

def main():
    # Configuration
    MODEL_NAME = 'stanford-crfm/music-large-800k'
    CHECKPOINT_PATH = os.path.join('training_output', 'checkpoint-best')
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_dir = Path("training_output/test_input")
    output_dir = Path("output_midi_reliable")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try loading model, but continue even if it fails
    model = None
    try:
        print(f"Loading base model from {MODEL_NAME}...")
        base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
        
        print(f"Loading fine-tuned adapter from {CHECKPOINT_PATH}...")
        model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
        model.to(DEVICE)
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Warning: Failed to load model: {e}")
        print("Continuing with direct conversion only.")
    
    # Process all files
    midi_files = list(input_dir.glob("*.mid"))
    print(f"\nProcessing {len(midi_files)} MIDI files...")
    
    success_count = 0
    for input_path in tqdm(midi_files):
        output_path = output_dir / f"{input_path.stem}_processed.mid"
        if process_midi_file(input_path, output_path, model):
            success_count += 1
    
    print(f"\nSuccessfully processed {success_count}/{len(midi_files)} files!")
    print(f"Output files are saved in {output_dir}")

if __name__ == "__main__":
    main()