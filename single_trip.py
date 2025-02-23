import os
import torch
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
from anticipation.convert import midi_to_events, events_to_midi

# Directories for individual file checking
INPUT_DIR = 'dataset/input'
TARGET_DIR = 'dataset/target'

def tensor_to_midi(tensor):
    """Convert a tensor of events to a MIDI file object."""
    events = tensor.tolist()
    
    # Debug: Inspect duration tokens (assuming every 5th token starting at index 1 is duration)
    if len(events) >= 2:
        duration_tokens = events[1::5]
        print("DEBUG: Duration tokens stats - min:", min(duration_tokens), "max:", max(duration_tokens))
    
    return events_to_midi(events)

def check_single_file(fname, directory):
    """Perform a round-trip conversion on a single file from a given directory."""
    try:
        file_path = os.path.join(directory, fname)
        # Convert MIDI file to event tokens.
        events = midi_to_events(file_path, debug=True)
        # Convert back to a MIDI object.
        tensor = torch.tensor(events)
        midi_obj = tensor_to_midi(tensor)
        print(f"Processed {fname} in {directory} with {len(events)} tokens")
        return True
    except Exception as e:
        print(f"Round trip failed for {fname} in {directory} due to error: {e}")
        return False

def process_directory(directory):
    """Process every file in the given directory using a parallel executor."""
    files = os.listdir(directory)
    total = len(files)
    failed = 0

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(check_single_file, fname, directory): fname for fname in files}
        for future in tqdm(concurrent.futures.as_completed(futures), total=total, desc=f"Processing {directory}"):
            if not future.result():
                failed += 1
    return total, failed

def main():
    total_input, failed_input = process_directory(INPUT_DIR)
    total_target, failed_target = process_directory(TARGET_DIR)
    
    print(f"\nINPUT DIRECTORY: Total files = {total_input}, Failures = {failed_input}")
    print(f"TARGET DIRECTORY: Total files = {total_target}, Failures = {failed_target}")

if __name__ == "__main__":
    main()