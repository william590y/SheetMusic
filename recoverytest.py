from train import MIDIPairDataset, tensor_to_midi
from anticipation.convert import midi_to_events, events_to_midi
import os

INPUT_DIR = os.path.join(os.getcwd(), "dataset", "input")
TARGET_DIR = os.path.join(os.getcwd(), "dataset", "target")

def unpad(padded_tensor, pad_value=-100):
    # Find the index of the first pad token (if any) and slice to that length.
    non_pad_indices = (padded_tensor != pad_value).nonzero(as_tuple=True)[0]
    if non_pad_indices.numel() > 0:
        last_index = non_pad_indices[-1].item() + 1
    else:
        last_index = padded_tensor.size(0)
    return padded_tensor[:last_index]

def main():
    dataset = MIDIPairDataset(INPUT_DIR, TARGET_DIR)
    print(f"Number of rejected MIDI pairs: {dataset.rejected}")

    recovery_dir = os.path.join(os.getcwd(), "recovery_test")
    os.makedirs(recovery_dir, exist_ok=True)

    # Load and process first 10 samples from dataset
    for i, (input_tensor, target_tensor) in enumerate(dataset):
        if i >= 10:
            break
        target_tensor = unpad(input_tensor)
        midi = tensor_to_midi(target_tensor)
        midi_path = os.path.join(recovery_dir, f"recovered_{i}.mid")
        midi.save(midi_path)
        print(f"Saved recovered midi: {midi_path}")

if __name__ == "__main__":
    main()