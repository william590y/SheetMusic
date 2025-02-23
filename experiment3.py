import os
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from anticipation.convert import midi_to_events, events_to_midi  # ...existing code...

def tensor_to_midi(tensor):
    """Convert a tensor of events to a MIDI file object"""
    events = tensor.cpu().int().tolist()
    # Remove any padding (zeros at the end)
    events = [e for e in events if e != 0]
    # Ensure length is divisible by 3 (trim if needed)
    # while len(events) % 3 != 0:
    #    events = events[:-1]

    if len(events) >= 2:
        duration_tokens = events[1::5]
        print("DEBUG: Duration tokens stats - min:", min(duration_tokens), "max:", max(duration_tokens))
    
    return events_to_midi(events)

class SingleMIDIDataset(Dataset):
    def __init__(self, midi_file):
        if not Path(midi_file).exists():
            raise FileNotFoundError(f"File not found: {midi_file}")
        self.midi_file = midi_file
        # Convert MIDI file to event tokens
        self.events = midi_to_events(midi_file, debug=True)
        
    def __len__(self):
        return 1  # single sample
    
    def __getitem__(self, idx):
        return torch.tensor(self.events)

def main():
    # Path to the test input MIDI file (ensure it exists in current working directory)
    midi_path = "test_input.mid"
    
    # Initialize dataset and dataloader (default collate_fn)
    dataset = SingleMIDIDataset(midi_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Iterate through the dataloader and recover MIDI files
    output_dir = Path("experiment3_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, batch in enumerate(loader):
        # batch shape: (batch_size, sequence_length)
        sample_tensor = batch[0]
        raw_tokens = sample_tensor.cpu().int().tolist()
        print("First 300 tokens:", raw_tokens[:300])
        recovered_midi = tensor_to_midi(sample_tensor)
        output_file = output_dir / f"test_input_dataloader_{idx}.mid"
        recovered_midi.save(str(output_file))
        print(f"Saved recovered MIDI file to: {output_file}")

if __name__ == "__main__":
    main()
