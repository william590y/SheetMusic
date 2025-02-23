import os
import torch
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
from anticipation.convert import midi_to_events, events_to_midi

def test_tensor_conversion(file_path):
    """Test tensor conversion for a single file."""
    try:
        # Convert MIDI to events
        original_events = midi_to_events(file_path, debug=True)
        
        # Convert to tensor
        tensor = torch.tensor(original_events)
        
        # Convert back to list
        recovered_events = tensor.tolist()
        
        # Check equality
        are_equal = original_events == recovered_events
        
        # Test MIDI conversion
        original_midi = events_to_midi(original_events)
        recovered_midi = events_to_midi(recovered_events)
        
        return {
            'filename': os.path.basename(file_path),
            'success': True,
            'equal': are_equal,
            'length': len(original_events)
        }
        
    except Exception as e:
        return {
            'filename': os.path.basename(file_path),
            'success': False,
            'error': str(e),
            'equal': False,
            'length': 0
        }

def main():
    target_dir = 'dataset/target'
    files = [os.path.join(target_dir, f) for f in os.listdir(target_dir)]
    
    success_count = 0
    equal_count = 0
    total_files = len(files)
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(test_tensor_conversion, f): f for f in files}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=total_files):
            result = future.result()
            if result['success']:
                success_count += 1
                if result['equal']:
                    equal_count += 1
            else:
                print(f"\nFailed on {result['filename']}: {result.get('error', 'Unknown error')}")
    
    print(f"\nResults:")
    print(f"Total files processed: {total_files}")
    print(f"Successful conversions: {success_count}")
    print(f"Failed conversions: {total_files - success_count}")
    print(f"Equal after round trip: {equal_count}")
    print(f"Changed after round trip: {success_count - equal_count}")

if __name__ == "__main__":
    main()