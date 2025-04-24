import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def analyze_traffic_flow(folder_path, partition_type):
    """
    Analyze traffic flow from Excel files in a folder and its subfolders.
    
    Args:
        folder_path (str): Path to the folder containing Excel files
        partition_type (int): 1 for distraction-based partition, 2 for distribution-based partition
    """
    # Find all Excel files in the folder and subfolders
    excel_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.xlsx', '.xls')):
                excel_files.append(os.path.join(root, file))
    
    if not excel_files:
        print(f"No Excel files found in {folder_path}")
        return
    
    print(f"Found {len(excel_files)} Excel files")
    
    # Process each Excel file
    all_data = []
    for file_path in excel_files:
        try:
            # Read the Excel file
            df = pd.read_excel(file_path, sheet_name="Detailed Results")
            
            # Extract the file name for the legend
            file_name = os.path.basename(file_path)
            
            # Add a column for the file source
            df['File'] = file_name
            
            # Append to the list of all data
            all_data.append(df)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    if not all_data:
        print("No valid data found in the Excel files")
        return
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    if partition_type == 1:  # Distraction-based partition
        # Get unique distraction percentages
        distraction_percentages = combined_data['Percentage of Distracted Vehicles'].unique()
        distraction_percentages.sort()
        
        # Color map
        colors = plt.cm.rainbow(pd.np.linspace(0, 1, len(distraction_percentages)))
        
        # Plot each distraction percentage with a different color
        for i, distraction in enumerate(distraction_percentages):
            subset = combined_data[combined_data['Percentage of Distracted Vehicles'] == distraction]
            plt.scatter(subset['Density'], subset['Flow'], 
                       color=colors[i], 
                       label=f"Distraction: {distraction}%",
                       alpha=0.7)
        
        plt.title('Flow vs. Density by Distraction Percentage')
        
    elif partition_type == 2:  # Distribution-based partition (for future implementation)
        # For now, just plot by file as a placeholder
        files = combined_data['File'].unique()
        colors = plt.cm.rainbow(pd.np.linspace(0, 1, len(files)))
        
        for i, file in enumerate(files):
            subset = combined_data[combined_data['File'] == file]
            plt.scatter(subset['Density'], subset['Flow'], 
                       color=colors[i], 
                       label=f"File: {file}",
                       alpha=0.7)
        
        plt.title('Flow vs. Density by File Distribution')
    
    plt.xlabel('Density (vehicles/km/lane)')
    plt.ylabel('Flow (vehicles/hour)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    output_file = os.path.join(folder_path, f"flow_density_analysis_type{partition_type}.png")
    plt.savefig(output_file)
    print(f"Figure saved to {output_file}")
    
    # Show the figure
    plt.show()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze traffic flow from simulation Excel files.')
    parser.add_argument('folder_path', type=str, help='Path to folder containing Excel files')
    parser.add_argument('--partition', type=int, choices=[1, 2], default=1,
                        help='Partition type: 1 for distraction-based, 2 for distribution-based')
    
    args = parser.parse_args()
    
    # Run the analysis
    analyze_traffic_flow(args.folder_path, args.partition)

if __name__ == "__main__":
    main()