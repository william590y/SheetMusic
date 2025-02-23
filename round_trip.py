import os
import torch
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
from anticipation.convert import midi_to_events, events_to_midi
import traceback  # new import

# Directories for the paired MIDI files (adjust paths if needed)
INPUT_DIR = 'dataset/input'
TARGET_DIR = 'dataset/target'

def tensor_to_midi(tensor):
    """
    Convert a tensor of events to a MIDI file object.
    """
    events = tensor.tolist()
    
    # Debug: Inspect tokens (assuming every 5th token starting at index 1 is a duration token)
    if len(events) >= 2:
        duration_tokens = events[1::5]
        print("DEBUG: Duration tokens stats - min:", min(duration_tokens), "max:", max(duration_tokens))
    
    return events_to_midi(events)

def check_round_trip(fname, input_dir, target_dir):
    """
    Check round-trip convertibility for a pair of files.
    Returns True if both conversions succeed, False otherwise.
    """
    try:
        input_path = os.path.join(input_dir, fname)
        target_path = os.path.join(target_dir, fname)
        
        # Convert MIDI files to event tokens.
        input_events = midi_to_events(input_path, debug=True)
        target_events = midi_to_events(target_path, debug=True)
        
        # Attempt round-trip conversion.
        input_midi_obj = tensor_to_midi(torch.tensor(input_events))
        target_midi_obj = tensor_to_midi(torch.tensor(target_events))
        
        print(f"Processed {fname}: Input tokens={len(input_events)}; Target tokens={len(target_events)}")
        return True
    except Exception as e:
        print(f"Round trip failed for {fname} due to error: {e}\n{traceback.format_exc()}")
        return False

def main():
    input_files = set(os.listdir(INPUT_DIR))
    target_files = set(os.listdir(TARGET_DIR))
    common_files = input_files & target_files

    if not common_files:
        raise ValueError("No common filenames found between input and target directories.")
    
    total = len(common_files)
    failed = 0

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(check_round_trip, fname, INPUT_DIR, TARGET_DIR): fname for fname in common_files}
        for future in tqdm(concurrent.futures.as_completed(futures), total=total, desc="Checking round trips"):
            fname = futures[future]
            if not future.result():
                failed += 1
    
    print(f"\nTotal files checked: {total}")
    print(f"Number of files that failed round trip conversion: {failed}")

if __name__ == "__main__":
    main()