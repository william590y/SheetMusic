import os
import torch
from pathlib import Path
from anticipation.convert import midi_to_events, events_to_midi  # see [anticipation/convert.py](anticipation/convert.py)

def tensor_to_midi(tensor):
    """Convert a tensor of events to a MIDI file object"""
    # Ensure tensor is on CPU and convert to integer list
    events = tensor.cpu().int().tolist()
    
    # Remove any padding (zeros at the end)
    events = [e for e in events if e != 0]
    
    # Ensure length is divisible by 3
    while len(events) % 3 != 0:
        events = events[:-1]
        
    # Convert to MIDI using events_to_midi from anticipation.convert
    return events_to_midi(events)

def main():
    # Path to an example MIDI file
    midi_path = Path("test_input.mid")
    if not midi_path.exists():
        print(f"File not found: {midi_path}")
        return
    
    # Convert MIDI file to events tokens and then to tensor
    events = midi_to_events(str(midi_path), debug=True)
    tensor = torch.tensor(events)
    
    # Convert back to MIDI
    midi_object = tensor_to_midi(tensor)
    
    # Save the recovered MIDI file
    output_path = Path("recovered_experiment.mid")
    midi_object.save(str(output_path))
    print(f"Recovered MIDI file saved to: {output_path}")

if __name__ == "__main__":
    main()