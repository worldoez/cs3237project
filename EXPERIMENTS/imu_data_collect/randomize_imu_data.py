#!/usr/bin/env python3
"""
Randomize IMU data by shuffling action groups while keeping each group intact.

Usage:
    python randomize_imu_data.py file.csv
    python randomize_imu_data.py file.csv -o randomized_output.csv
    python randomize_imu_data.py file.csv --seed 42
"""

import pandas as pd
import sys
import argparse
import random
from pathlib import Path


def randomize_imu_data(input_file, output_file=None, seed=None):
    """
    Randomize IMU data by shuffling action groups.
    
    Args:
        input_file: Input CSV file path
        output_file: Output CSV file path (optional)
        seed: Random seed for reproducibility (optional)
    """
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        print(f"Using random seed: {seed}")
    
    # Read the CSV
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    # Verify required column exists
    if 'action_id' not in df.columns:
        print("Error: CSV does not have 'action_id' column")
        sys.exit(1)
    
    print(f"\nProcessing: {Path(input_file).name}")
    print(f"  Total rows: {len(df)}")
    print(f"  Unique action_ids: {df['action_id'].nunique()}")
    print(f"  Action_id range: {df['action_id'].min()} to {df['action_id'].max()}")
    
    # Group by action_id
    grouped = df.groupby('action_id')
    
    # Get list of action_ids
    action_ids = list(grouped.groups.keys())
    original_order = action_ids.copy()
    
    # Shuffle the action_ids
    random.shuffle(action_ids)
    
    print(f"\nShuffling {len(action_ids)} action groups...")
    
    # Collect shuffled groups
    shuffled_groups = []
    for new_id, old_id in enumerate(action_ids, start=1):
        group = grouped.get_group(old_id).copy()
        # Renumber action_id to maintain sequential order
        group['action_id'] = new_id
        shuffled_groups.append(group)
    
    # Combine all groups
    result = pd.concat(shuffled_groups, ignore_index=True)
    
    # Determine output filename
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.stem + '_randomized' + input_path.suffix
    
    # Save to output file
    result.to_csv(output_file, index=False)
    
    # Show sample of the mapping
    print("\nSample of action_id remapping (old -> new):")
    for i in range(min(5, len(action_ids))):
        old_id = action_ids[i]
        new_id = i + 1
        action_label = grouped.get_group(old_id)['action_label'].iloc[0]
        row_count = len(grouped.get_group(old_id))
        print(f"  {old_id} -> {new_id} ({action_label}, {row_count} rows)")
    
    if len(action_ids) > 5:
        print(f"  ... and {len(action_ids) - 5} more")
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"✓ Successfully randomized data")
    print(f"  Output saved to: {output_file}")
    print(f"  Total rows: {len(result)}")
    print(f"  Action groups shuffled: {len(action_ids)}")
    print("=" * 60)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Randomize IMU data by shuffling action groups',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python randomize_imu_data.py data.csv
  python randomize_imu_data.py data.csv -o shuffled_data.csv
  python randomize_imu_data.py data.csv --seed 42
        """
    )
    
    parser.add_argument('file', help='Input CSV file to randomize')
    parser.add_argument('-o', '--output', default=None,
                        help='Output file name (default: <input>_randomized.csv)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Verify file exists
    if not Path(args.file).exists():
        print(f"Error: File not found: {args.file}")
        sys.exit(1)
    
    randomize_imu_data(args.file, args.output, args.seed)


if __name__ == '__main__':
    main()