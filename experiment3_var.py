import os
import torch
import concurrent.futures
from pathlib import Path
from anticipation.convert import midi_to_events, events_to_midi

def tensor_to_midi(tensor):
    """Convert a tensor of events to a MIDI file object using the experiment3 pipeline."""
    # Convert tensor to list of integers and remove padding (zeros)
    events = tensor.cpu().int().tolist()
    events = [e for e in events if e != 0]
    
    # Debug: inspect duration tokens (assumes every 5th token starting at index 1 is duration)
    if len(events) >= 2:
        duration_tokens = events[1::5]
        print("DEBUG: Duration tokens stats - min:", min(duration_tokens), "max:", max(duration_tokens))
    
    return events_to_midi(events)

def process_file(fname, input_dir, output_dir):
    # Full path to the input MIDI file.
    input_path = os.path.join(input_dir, fname)
    
    # Convert MIDI to event tokens (using experiment3 logic)
    events = midi_to_events(input_path, debug=True)
    sample_tensor = torch.tensor(events)
    
    # Convert events back to a MIDI object.
    recovered_midi = tensor_to_midi(sample_tensor)
    
    # Save the recovered MIDI file in the output directory.
    out_file = os.path.join(output_dir, fname)
    recovered_midi.save(out_file)
    print(f"Saved recovered MIDI file to: {out_file}")

def main():
    # Directories: same as experiment4
    input_dir = 'dataset/target'
    output_dir = 'experiment3_output'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get all MIDI filenames from the input directory
    files = os.listdir(input_dir)
    if not files:
        raise ValueError(f"No files found in {input_dir}")
    
    # Process files concurrently using a ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file, fname, input_dir, output_dir) for fname in files]
        for future in concurrent.futures.as_completed(futures):
            # Propagate any exceptions
            future.result()

if __name__ == "__main__":
    main()