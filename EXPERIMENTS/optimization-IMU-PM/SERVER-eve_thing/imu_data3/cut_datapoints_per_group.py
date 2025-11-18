import pandas as pd
import sys
import os

def cut_datapoints_per_group(input_file, output_file, target_points):
    """
    Cut datapoints from start and end of each action group to keep only the center.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        target_points: Target number of datapoints to keep in center of each group
    """
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    print(f"Columns found: {list(df.columns)}")
    print(f"Total rows: {len(df)}\n")
    
    # Find the action_id column (check for common variations)
    action_col = None
    for col in df.columns:
        if 'action' in col.lower() and 'id' in col.lower():
            action_col = col
            break
    
    if action_col is None:
        # If no action_id found, use the first column
        action_col = df.columns[0]
        print(f"Warning: No 'action_id' column found. Using first column: '{action_col}'")
    else:
        print(f"Using column '{action_col}' as action identifier")
    
    print()
    
    # Group by action_id
    grouped = df.groupby(action_col)
    
    processed_groups = []
    
    for action_id, group in grouped:
        group = group.reset_index(drop=True)
        current_size = len(group)
        
        print(f"Processing action_id {action_id}: {current_size} points -> {target_points} points")
        
        if current_size == target_points:
            # Already at target
            processed_groups.append(group)
            print(f"  Already exactly {target_points} points - keeping all")
            continue
        
        elif current_size < target_points:
            # Cannot cut if we need more points than available
            processed_groups.append(group)
            print(f"  WARNING: Group has fewer points ({current_size}) than target ({target_points})")
            print(f"  Keeping all {current_size} points (cannot expand)")
            continue
        
        else:
            # Need to cut from start and end (current_size > target_points)
            points_to_remove = current_size - target_points
            
            # Calculate how much to remove from each end
            remove_start = points_to_remove // 2
            remove_end = points_to_remove - remove_start
            
            # Calculate the indices to keep (the center portion)
            start_idx = remove_start
            end_idx = current_size - remove_end
            
            # Extract the center portion
            group = group.iloc[start_idx:end_idx].reset_index(drop=True)
            
            print(f"  Removed {remove_start} from start and {remove_end} from end")
            print(f"  Kept center portion: indices {start_idx} to {end_idx-1} ({len(group)} points)")
        
        processed_groups.append(group)
    
    # Combine all processed groups
    result_df = pd.concat(processed_groups, ignore_index=True)
    
    # Save to output file
    result_df.to_csv(output_file, index=False)
    
    print(f"\nProcessing complete! Saved to {output_file}")
    print(f"Total rows: {len(result_df)}")
    
    # Print summary statistics
    print("\nSummary by action:")
    summary = result_df.groupby(action_col).size()
    for action_id, count in summary.items():
        print(f"  Action {action_id}: {count} points")

# Main execution
if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) < 3:
        print("Usage: python cut_datapoints_per_group.py <input_file> <target_points> [output_file]")
        print("  input_file: Path to input CSV file (required)")
        print("  target_points: Target number of datapoints to keep in center (required)")
        print("  output_file: Path to output CSV file (optional, default: cut_data.csv)")
        print("\nExamples:")
        print("  python cut_datapoints_per_group.py data.csv 150")
        print("  python cut_datapoints_per_group.py data.csv 150 trimmed.csv")
        sys.exit(1)
    
    # Get input file (required)
    input_file = sys.argv[1]
    
    # Get target points (required)
    try:
        target_points = int(sys.argv[2])
    except ValueError:
        print(f"Error: target_points must be an integer, got '{sys.argv[2]}'")
        sys.exit(1)
    
    if target_points <= 0:
        print(f"Error: target_points must be positive, got {target_points}")
        sys.exit(1)
    
    # Get output file (optional)
    if len(sys.argv) >= 4:
        output_file = sys.argv[3]
    else:
        # Generate default output filename in the same directory as the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(script_dir, "cut_data.csv")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Target points per group: {target_points}")
    print("=" * 60 + "\n")
    
    cut_datapoints_per_group(input_file, output_file, target_points)