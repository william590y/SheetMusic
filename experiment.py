import os
from anticipation.convert import midi_to_events, events_to_midi
from pathlib import Path

def main():
    # Path to an example MIDI file that exists in your dataset.
    # You might copy an example file into the project folder or adjust this path.
    midi_file = "test_input.mid"
    
    if not os.path.exists(midi_file):
        print(f"Example MIDI file not found at {midi_file}. Please provide a valid file.")
        return

    # Convert the original MIDI file to events tokens.
    print("Converting original MIDI to events...")
    original_tokens = midi_to_events(midi_file, debug=True)
    print(f"Number of original tokens: {len(original_tokens)}")

    # Convert events back to a MIDI object.
    print("Converting events back to MIDI...")
    midi_object = events_to_midi(original_tokens, debug=True)
    print(type(midi_object))

    
    # Save the re-converted MIDI file.
    output_file = "experiment_output.mid"
    midi_object.save(str(Path(output_file)))
    print(f"Re-converted MIDI file saved to: {output_file}")

    # Convert the re-converted MIDI file back to events.
    print("Converting re-converted MIDI back to events...")
    new_tokens = midi_to_events(output_file, debug=True)
    print(f"Number of new tokens: {len(new_tokens)}")

    # Compare original and new token sequences.
    if original_tokens == new_tokens:
        print("The conversion functions appear to be perfect inverses (unexpected).")
    else:
        print("The conversions are not perfect inverses, as suspected.")
        # Optionally, print out a summary of differences.
        diff = abs(len(original_tokens) - len(new_tokens))
        print(f"Token count difference: {diff}")

if __name__ == "__main__":
    main()