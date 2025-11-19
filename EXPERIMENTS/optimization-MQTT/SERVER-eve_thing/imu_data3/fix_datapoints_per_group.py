import pandas as pd
import random
import sys
import os

def clean_csv_data(input_file, output_file, target_points=150):
    """
    Clean CSV data by reducing each action group to exactly target_points datapoints.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        target_points: Target number of datapoints per group (default: 150)
    """
    # Read the CSV file (comma-separated)
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
    
    cleaned_groups = []
    
    for action_id, group in grouped:
        group = group.reset_index(drop=True)
        current_size = len(group)
        
        print(f"Processing action_id {action_id}: {current_size} points -> {target_points} points")
        
        if current_size == target_points:
            # Already at target
            cleaned_groups.append(group)
            print(f"  Already exactly {target_points} points")
            continue
        
        elif current_size < target_points:
            # Need to duplicate datapoints
            needed = target_points - current_size
            print(f"  Need to add {needed} points by duplication")
            
            # Randomly select indices to duplicate
            indices_to_duplicate = random.choices(range(current_size), k=needed)
            
            # Create duplicates
            duplicates = group.iloc[indices_to_duplicate].reset_index(drop=True)
            
            # Combine original and duplicates
            group = pd.concat([group, duplicates], ignore_index=True)
            
            # Shuffle to distribute duplicates randomly
            group = group.sample(frac=1, random_state=random.randint(0, 10000)).reset_index(drop=True)
            
            print(f"  After duplication and shuffling: {len(group)} points")
        
        else:
            # Need to reduce datapoints (current_size > target_points)
            # Step 1: Random downsampling with varying jumps (3, 4, or 5)
            # But only remove enough to get close to target
            indices_to_remove = set()
            i = 0
            
            # Randomly choose jump size: 3, 4, or 5
            jump = random.choice([3, 4, 5])
            
            while i < current_size and len(group) - len(indices_to_remove) > target_points:
                # Mark this index for removal
                if i < current_size:
                    indices_to_remove.add(i)
                
                # Move to next position with random jump
                i += jump
                
                # Occasionally change the jump size
                if random.random() < 0.3:
                    jump = random.choice([3, 4, 5])
            
            # Keep indices that weren't marked for removal
            indices_to_keep = [i for i in range(current_size) if i not in indices_to_remove]
            group = group.iloc[indices_to_keep].reset_index(drop=True)
            
            print(f"  After random jump removal (jump size {jump}): {len(group)} points")
            
            # Step 2: If still not exactly target_points, trim equally from both ends
            if len(group) > target_points:
                excess = len(group) - target_points
                
                # Calculate how much to remove from each end
                remove_start = excess // 2
                remove_end = excess - remove_start
                
                # Trim from both ends
                if remove_end > 0:
                    group = group.iloc[remove_start:-remove_end].reset_index(drop=True)
                else:
                    group = group.iloc[remove_start:].reset_index(drop=True)
                
                print(f"  After trimming {remove_start} from start and {remove_end} from end: {len(group)} points")
        
        cleaned_groups.append(group)
    
    # Combine all cleaned groups
    result_df = pd.concat(cleaned_groups, ignore_index=True)
    
    # Save to output file with comma separator
    result_df.to_csv(output_file, index=False)
    
    print(f"\nCleaning complete! Saved to {output_file}")
    print(f"Total rows: {len(result_df)}")
    
    # Print summary statistics
    print("\nSummary by action:")
    summary = result_df.groupby(action_col).size()
    for action_id, count in summary.items():
        print(f"  Action {action_id}: {count} points")

# Example usage
if __name__ == "__main__":
    # Set random seed for reproducibility (optional)
    random.seed(42)
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python clean_csv.py <input_file> [output_file] [target_points]")
        print("  input_file: Path to input CSV file (required)")
        print("  output_file: Path to output CSV file (optional, default: cleaned_data.csv in script directory)")
        print("  target_points: Target number of datapoints per group (optional, default: 150)")
        print("\nExamples:")
        print("  python clean_csv.py data.csv")
        print("  python clean_csv.py data.csv cleaned.csv")
        print("  python clean_csv.py data.csv cleaned.csv 200")
        sys.exit(1)
    
    # Get input file (required)
    input_file = sys.argv[1]
    
    # Get output file (optional)
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        # Generate default output filename in the same directory as the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(script_dir, "cleaned_data.csv")
    
    # Get target points (optional)
    if len(sys.argv) >= 4:
        try:
            target_points = int(sys.argv[3])
        except ValueError:
            print(f"Error: target_points must be an integer, got '{sys.argv[3]}'")
            sys.exit(1)
    else:
        target_points = 150
    
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Target points per group: {target_points}")
    print("=" * 60 + "\n")
    
    clean_csv_data(input_file, output_file, target_points)