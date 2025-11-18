import pandas as pd
import sys
import os
import random

def remove_random_label_groups(input_file, output_file, action_label, num_groups_to_remove):
    """
    Remove random groups (by action_id) that match a specific action_label.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        action_label: The action label to filter (e.g., 'straight', 'jump', 'right')
        num_groups_to_remove: Number of random groups with that label to remove
    """
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    print(f"Columns found: {list(df.columns)}")
    print(f"Total rows: {len(df)}\n")
    
    # Check if required columns exist
    if 'action_id' not in df.columns:
        print("Error: 'action_id' column not found in CSV")
        sys.exit(1)
    
    if 'action_label' not in df.columns:
        print("Error: 'action_label' column not found in CSV")
        sys.exit(1)
    
    # Find all unique action_ids that have the specified action_label
    matching_groups = df[df['action_label'] == action_label]['action_id'].unique()
    
    print(f"Action label to remove: '{action_label}'")
    print(f"Total groups with label '{action_label}': {len(matching_groups)}")
    print(f"Groups to remove: {num_groups_to_remove}\n")
    
    if len(matching_groups) == 0:
        print(f"Warning: No groups found with action_label '{action_label}'")
        print("No changes made. Copying input to output.")
        df.to_csv(output_file, index=False)
        return
    
    if num_groups_to_remove > len(matching_groups):
        print(f"Warning: Requested to remove {num_groups_to_remove} groups, but only {len(matching_groups)} exist.")
        print(f"Removing all {len(matching_groups)} groups with label '{action_label}'")
        num_groups_to_remove = len(matching_groups)
    
    # Randomly select groups to remove
    groups_to_remove = random.sample(list(matching_groups), num_groups_to_remove)
    
    print(f"Randomly selected action_ids to remove: {sorted(groups_to_remove)}\n")
    
    # Show details of groups being removed
    for action_id in sorted(groups_to_remove):
        group_size = len(df[df['action_id'] == action_id])
        print(f"  Removing action_id {action_id}: {group_size} datapoints")
    
    # Filter out the selected groups
    result_df = df[~df['action_id'].isin(groups_to_remove)]
    
    # Renumber action_ids to be sequential (1, 2, 3, ...)
    print(f"\nRenumbering action_ids to be sequential...")
    unique_action_ids = result_df['action_id'].unique()
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(unique_action_ids), start=1)}
    result_df['action_id'] = result_df['action_id'].map(id_mapping)
    
    # Save to output file
    result_df.to_csv(output_file, index=False)
    
    print(f"\nProcessing complete! Saved to {output_file}")
    print(f"Removed {len(df) - len(result_df)} rows")
    print(f"Remaining rows: {len(result_df)}")
    
    # Print summary statistics
    print("\nRemaining groups by action_label:")
    summary = result_df.groupby('action_label')['action_id'].nunique()
    for label, count in summary.items():
        print(f"  {label}: {count} groups")

# Main execution
if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) < 4:
        print("Usage: python remove_n_label_groups.py <action_label> <num_groups> <input_file> [output_file]")
        print("  action_label: The action label to filter (e.g., 'straight', 'jump', 'right') (required)")
        print("  num_groups: Number of random groups with that label to remove (required)")
        print("  input_file: Path to input CSV file (required)")
        print("  output_file: Path to output CSV file (optional, default: removed_data.csv)")
        print("\nExamples:")
        print("  python remove_n_label_groups.py straight 1 data.csv")
        print("  python remove_n_label_groups.py jump 2 input.csv output.csv")
        sys.exit(1)
    
    # Get action label (required)
    action_label = sys.argv[1]
    
    # Get number of groups to remove (required)
    try:
        num_groups_to_remove = int(sys.argv[2])
    except ValueError:
        print(f"Error: num_groups must be an integer, got '{sys.argv[2]}'")
        sys.exit(1)
    
    if num_groups_to_remove <= 0:
        print(f"Error: num_groups must be positive, got {num_groups_to_remove}")
        sys.exit(1)
    
    # Get input file (required)
    input_file = sys.argv[3]
    
    # Get output file (optional)
    if len(sys.argv) >= 5:
        output_file = sys.argv[4]
    else:
        # Generate default output filename in the same directory as the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(script_dir, "removed_data.csv")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print("=" * 60 + "\n")
    
    remove_random_label_groups(input_file, output_file, action_label, num_groups_to_remove)