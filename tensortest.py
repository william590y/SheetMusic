import os
import torch
from anticipation.convert import midi_to_events, events_to_midi

# Get the first MIDI file from the target directory
target_dir = 'dataset/target'
first_midi = os.listdir(target_dir)[152]
file_path = os.path.join(target_dir, first_midi)

# Convert MIDI to events
original_events = midi_to_events(file_path, debug=True)
print("\nOriginal events:")
print(f"Type: {type(original_events)}")
print(f"Length: {len(original_events)}")
print(f"First 10 events: {original_events[:10]}")
print(f"Types of first 10: {[type(x) for x in original_events[:10]]}")

# Convert to tensor
tensor = torch.tensor(original_events)
print("\nTensor info:")
print(f"Shape: {tensor.shape}")
print(f"dtype: {tensor.dtype}")

# Convert back to list
recovered_events = tensor.tolist()
print("\nRecovered events:")
print(f"Type: {type(recovered_events)}")
print(f"Length: {len(recovered_events)}")
print(f"First 10 events: {recovered_events[:10]}")
print(f"Types of first 10: {[type(x) for x in recovered_events[:10]]}")

# Check equality
are_equal = original_events == recovered_events
print(f"\nLists are equal: {are_equal}")

if not are_equal:
    # Find first difference
    for i, (orig, rec) in enumerate(zip(original_events, recovered_events)):
        if orig != rec:
            print(f"\nFirst difference at index {i}:")
            print(f"Original: {orig} (type: {type(orig)})")
            print(f"Recovered: {rec} (type: {type(rec)})")
            break

# Test MIDI conversion
print("\nTesting MIDI conversion:")
try:
    print("Converting original events to MIDI...")
    original_midi = events_to_midi(original_events)
    print("Success!")
except Exception as e:
    print(f"Failed to convert original events: {e}")

try:
    print("Converting recovered events to MIDI...")
    recovered_midi = events_to_midi(recovered_events)
    print("Success!")
except Exception as e:
    print(f"Failed to convert recovered events: {e}")