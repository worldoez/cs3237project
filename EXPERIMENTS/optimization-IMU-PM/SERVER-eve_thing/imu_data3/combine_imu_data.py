#!/usr/bin/env python3
"""
Combine multiple IMU CSV files with unique action_ids across all files.

Usage:
    python combine_imu_data.py file1.csv file2.csv file3.csv ... fileX.csv
    python combine_imu_data.py *.csv
    python combine_imu_data.py data/*.csv -o combined_output.csv
"""

import pandas as pd
import sys
import argparse
from pathlib import Path


def combine_imu_files(input_files, output_file='combined_imu_data.csv'):
    """
    Combine multiple IMU CSV files with renumbered action_ids.
    
    Args:
        input_files: List of CSV file paths
        output_file: Output CSV file path
    """
    if not input_files:
        print("Error: No input files provided")
        sys.exit(1)
    
    combined_data = []
    max_action_id = 0
    
    print(f"Processing {len(input_files)} files...\n")
    
    for i, file in enumerate(input_files, 1):
        try:
            # Read the CSV
            df = pd.read_csv(file)
            
            # Verify required column exists
            if 'action_id' not in df.columns:
                print(f"Warning: '{file}' does not have 'action_id' column. Skipping.")
                continue
            
            original_min = df['action_id'].min()
            original_max = df['action_id'].max()
            
            # Offset the action_id by the current max
            df['action_id'] = df['action_id'] + max_action_id
            
            # Update max_action_id for the next file
            max_action_id = df['action_id'].max()
            
            # Add to combined data
            combined_data.append(df)
            
            print(f"[{i}/{len(input_files)}] {Path(file).name}")
            print(f"    Original action_ids: {original_min} to {original_max}")
            print(f"    New action_ids:      {df['action_id'].min()} to {df['action_id'].max()}")
            print(f"    Rows: {len(df)}\n")
            
        except Exception as e:
            print(f"Error processing '{file}': {e}")
            continue
    
    if not combined_data:
        print("Error: No valid data to combine")
        sys.exit(1)
    
    # Concatenate all dataframes
    result = pd.concat(combined_data, ignore_index=True)
    
    # Save to output file
    result.to_csv(output_file, index=False)
    
    # Print summary
    print("=" * 60)
    print(f"✓ Successfully combined {len(combined_data)} files")
    print(f"  Total rows: {len(result)}")
    print(f"  Final action_id range: {result['action_id'].min()} to {result['action_id'].max()}")
    print(f"  Unique action_ids: {result['action_id'].nunique()}")
    print(f"  Output saved to: {output_file}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Combine multiple IMU CSV files with unique action_ids',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python combine_imu_data.py file1.csv file2.csv file3.csv
  python combine_imu_data.py data/*.csv
  python combine_imu_data.py *.csv -o output.csv
        """
    )
    
    parser.add_argument('files', nargs='+', help='Input CSV files to combine')
    parser.add_argument('-o', '--output', default='combined_imu_data.csv',
                        help='Output file name (default: combined_imu_data.csv)')
    
    args = parser.parse_args()
    
    # Verify files exist
    valid_files = []
    for file in args.files:
        if Path(file).exists():
            valid_files.append(file)
        else:
            print(f"Warning: File not found: {file}")
    
    if not valid_files:
        print("Error: No valid input files found")
        sys.exit(1)
    
    combine_imu_files(valid_files, args.output)


if __name__ == '__main__':
    main()