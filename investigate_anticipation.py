import sys
import importlib
import inspect
import torch
import os
from pathlib import Path
from anticipation import convert
from anticipation.config import MAX_DUR

def print_function_source(func):
    """Print the source code of a function"""
    print(f"\n{'-'*40}")
    print(f"Source code for {func.__name__}:")
    print(f"{'-'*40}")
    print(inspect.getsource(func))
    print(f"{'-'*40}\n")

def examine_midi_file(midi_path):
    """Examine a MIDI file and print details of its event representation"""
    try:
        print(f"Examining MIDI file: {midi_path}")
        events = convert.midi_to_events(midi_path, debug=True)
        print(f"Number of events: {len(events)}")
        print(f"Event value range: min={min(events)}, max={max(events)}")
        
        # Analyze the event structure
        print("\nEvent structure analysis:")
        
        # Check if events follow the 5-token pattern
        if len(events) % 5 == 0:
            print("Events appear to follow a 5-token pattern")
            # Sample a few event groups
            for i in range(0, min(25, len(events)), 5):
                print(f"Event {i//5}: {events[i:i+5]}")
        else:
            print("Events do not follow a clean 5-token pattern")
            print("First 30 tokens:", events[:30])
        
        # Attempt conversion back to MIDI
        print("\nTrying round-trip conversion...")
        try:
            midi_object = convert.events_to_midi(events)
            print("✅ Round-trip conversion successful!")
        except Exception as e:
            print(f"❌ Round-trip conversion failed: {e}")
            
        return events
    except Exception as e:
        print(f"Error examining MIDI file: {e}")
        return None

# Print the relevant conversion functions
print_function_source(convert.events_to_compound)
print_function_source(convert.compound_to_midi)
print_function_source(convert.events_to_midi)

# Find a test MIDI file to examine
input_dir = Path("training_output/test_input")
test_files = list(input_dir.glob("*.mid"))
if test_files:
    test_midi = str(test_files[0])
    events = examine_midi_file(test_midi)
    
    if events:
        # Let's see if we can identify the pattern in the events that's being expected
        print("\nAnalyzing token patterns...")
        
        # Check if tokens come in groups of 5
        if len(events) % 5 == 0:
            event_groups = [events[i:i+5] for i in range(0, len(events), 5)]
            
            # Analyze each position in the 5-token groups
            for pos in range(5):
                values = [group[pos] for group in event_groups]
                print(f"Position {pos} values: min={min(values)}, max={max(values)}, unique={len(set(values))}")
else:
    print("No MIDI files found in training_output/test_input directory")

# Additional investigation into token vocabulary
print("\nInvestigating model token vocabulary...")
try:
    # Check if we can access anticipated token vocabulary size
    vocab_info = getattr(convert, "VOCAB_SIZE", None)
    if vocab_info:
        print(f"Vocabulary size defined in convert module: {vocab_info}")
    else:
        print("No explicit vocabulary size defined in convert module")
        
    # Look for vocabulary constraints in the code
    max_token = getattr(convert, "MAX_TOKEN", None)
    if max_token:
        print(f"Maximum token value: {max_token}")
    else:
        print("No explicit maximum token value defined")
        
except Exception as e:
    print(f"Error investigating vocabulary: {e}")