import os
import torch
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
from anticipation.config import MAX_DUR

# Directories for your paired MIDI files
INPUT_DIR = 'dataset/input'
TARGET_DIR = 'dataset/target'

mult = 1  # tried 1, 3 and 5 neither worked

# Import conversion functions from anticipation.convert
from anticipation.convert import midi_to_events, events_to_midi

def tensor_to_midi(tensor):
    """Convert a tensor of events to a MIDI file object"""
    events = tensor.cpu().int().tolist()
    # Remove any padding (zeros at the end)
    events = [e for e in events if e != 0]
    # Uncomment the following if you want to enforce event length divisibility

    # Debug: Inspect tokens (assuming duration tokens at every 5th token starting from index 1)
    if len(events) >= 2:
        duration_tokens = events[1::5]
        print("DEBUG: Duration tokens stats - min:", min(duration_tokens), "max:", max(duration_tokens))

    return events_to_midi(events)

def process_file(fname, input_dir, target_dir):
    try:
        input_path = os.path.join(input_dir, fname)
        target_path = os.path.join(target_dir, fname)

        # Convert MIDI files to event tokens
        input_events = midi_to_events(input_path, debug=True)
        target_events = midi_to_events(target_path, debug=True)

        # Attempt a round-trip conversion: tokens --> MIDI and back
        # (This checks that tokenization and detokenization work without error.)
        input_midi_obj = tensor_to_midi(torch.tensor(input_events))
        target_midi_obj = tensor_to_midi(torch.tensor(target_events))

        print(f"Processed {fname} with {len(input_events)//mult} input tokens and {len(target_events)//mult} target tokens")
        print("Conversion successful")
        return torch.tensor(input_events), torch.tensor(target_events)
    except Exception as e:
        print(f"Rejecting {fname} due to error: {e}")
        return None

class MIDIPairDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir, target_dir):
        self.pairs = []
        self.rejected = 0

        input_files = set(os.listdir(input_dir))
        target_files = set(os.listdir(target_dir))
        common_files = input_files & target_files

        if not common_files:
            raise ValueError("No common filenames found between input and output directories")
        
        # Process files in parallel
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_file, fname, input_dir, target_dir) for fname in common_files]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing MIDI files"):
                result = future.result()
                if result is not None:
                    self.pairs.append(result)
                else:
                    self.rejected += 1

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        return self.pairs[idx]

def main():
    # Create the dataset using the same directories as before
    dataset = MIDIPairDataset(INPUT_DIR, TARGET_DIR)
    
    # Report the number of rejected MIDI file pairs
    print(f"Number of rejected MIDI file pairs: {dataset.rejected}")

    # For example, take the first pair from the dataset if any exist
    if len(dataset) == 0:
        print("No valid MIDI file pairs found.")
        return

    input_tensor, target_tensor = dataset[0]
    
    # Remove extra dimensions if present
    if input_tensor.dim() > 1:
        input_tensor = input_tensor[0]
    if target_tensor.dim() > 1:
        target_tensor = target_tensor[0]
    
    # Convert the tensors back to MIDI objects
    input_midi = tensor_to_midi(input_tensor)
    target_midi = tensor_to_midi(target_tensor)
    
    # Save the recovered MIDI files for inspection
    output_dir = os.path.join("experiment4_output")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    input_midi_path = os.path.join(output_dir, "recovered_input.mid")
    target_midi_path = os.path.join(output_dir, "recovered_target.mid")
    input_midi.save(input_midi_path)
    target_midi.save(target_midi_path)
    
    print(f"Recovered MIDI files saved to {output_dir}")

if __name__ == "__main__":
    main()