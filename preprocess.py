from pathlib import Path
import pandas as pd
import shutil
import os

ASAP_BASE_PATH = Path(r"C:\Users\willi\Desktop\asap-dataset")
OUTPUT_ROOT = Path('dataset')
PERFORMANCE_DIR = OUTPUT_ROOT / 'input'
SCORE_DIR = OUTPUT_ROOT / 'target'

PERFORMANCE_DIR.mkdir(parents=True, exist_ok=True)
SCORE_DIR.mkdir(parents=True, exist_ok=True)

metadata_path = ASAP_BASE_PATH / 'metadata.csv'
df = pd.read_csv(metadata_path)

score_counter = {}

# File organization function
def organize_midi_files(row):
    try:
        # Generate a common base name from composer and title
        base_name = f"{row['composer']}_{row['title']}"

        # Update counter to handle duplicates (multiple performances)
        count = score_counter.get(base_name, 0) + 1
        score_counter[base_name] = count

        # If there's more than one instance, append an index. Otherwise, use just the base name.
        if count == 1:
            common_filename = f"{base_name}.mid"
        else:
            common_filename = f"{base_name}_{count}.mid"

        # Process performance MIDI
        if pd.notna(row['midi_performance']):
            perf_src = ASAP_BASE_PATH / row['midi_performance']
            perf_dst = PERFORMANCE_DIR / common_filename
            shutil.copy(perf_src, perf_dst)
            
        # Process score MIDI
        if pd.notna(row['midi_score']):
            score_src = ASAP_BASE_PATH / row['midi_score']
            score_dst = SCORE_DIR / common_filename
            shutil.copy(score_src, score_dst)
            
    except Exception as e:
        print(f"Error processing {row['title']}: {str(e)}")

# Process all entries
df.apply(organize_midi_files, axis=1)

print(f"Organization complete: {len(list(PERFORMANCE_DIR.glob('*.mid')))} performances, {len(list(SCORE_DIR.glob('*.mid')))} scores")


