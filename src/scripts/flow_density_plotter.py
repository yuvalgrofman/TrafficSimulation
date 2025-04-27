import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import argparse


def analyze_excel_files(folder_path, partition_type="distraction"):
    """
    Analyze all Excel files in the given folder and subfolders.
    
    Args:
        folder_path (str): Path to the folder containing Excel files
        partition_type (str): Type of partition for analysis ('distraction' or 'distribution')
    
    Returns:
        dict: Dictionary containing analysis results
    """
    results = {}
    
    # Walk through all folders and subfolders
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.xlsx') or file.endswith('.xls'):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                
                try:
                    # Read the Excel file
                    workbook = pd.ExcelFile(file_path)
                    
                    # Try to read the "Summary Results" sheet first
                    if "Summary Results" in workbook.sheet_names:
                        df = pd.read_excel(workbook, sheet_name="Summary Results")
                    else:
                        # If not available, read the first sheet
                        df = pd.read_excel(workbook, sheet_name=0)
                    
                    # Check if required columns exist
                    required_columns = ["Density", "Flow", "Average Flow", "Standard Deviation of Flow", 
                                        "Percentage of Distracted Vehicles"]
                    
                    # Handle different possible column names
                    if "Average Flow" in df.columns:
                        flow_col = "Average Flow"
                        flow_std_col = "Standard Deviation of Flow"
                    else:
                        flow_col = "Flow"
                        flow_std_col = None  # Will calculate standard deviation later
                    
                    # For sheets with detailed results, we'll need to calculate statistics
                    if flow_std_col is None:
                        # Group by density and distraction percentage
                        if partition_type == "distraction":
                            grouped = df.groupby(["Density", "Percentage of Distracted Vehicles"])
                        else:  # distribution
                            grouped = df.groupby(["Density", "Distribution Type"])  # Adjust as needed
                        
                        # Calculate average and standard deviation for each group
                        agg_results = grouped.agg({
                            flow_col: ["mean", "std"]
                        }).reset_index()
                        
                        # Flatten the column hierarchy
                        agg_results.columns = ["_".join(col).strip("_") for col in agg_results.columns.values]
                        
                        # Rename columns for consistency
                        agg_results.rename(columns={
                            "Density_": "Density",
                            f"{flow_col}_mean": "Average Flow",
                            f"{flow_col}_std": "Standard Deviation of Flow"
                        }, inplace=True)
                        
                        if partition_type == "distraction":
                            agg_results.rename(columns={
                                "Percentage of Distracted Vehicles_": "Percentage of Distracted Vehicles"
                            }, inplace=True)
                        else:  # distribution
                            agg_results.rename(columns={
                                "Distribution Type_": "Distribution Type"
                            }, inplace=True)
                        
                        df = agg_results
                    
                    # Extract file name without extension for labeling
                    file_name = os.path.splitext(os.path.basename(file_path))[0]
                    
                    # Organize results by partition type
                    if partition_type == "distraction":
                        partition_col = "Percentage of Distracted Vehicles"
                    else:  # distribution
                        partition_col = "Distribution Type"
                    
                    # Extract unique partition values
                    for partition_value in df[partition_col].unique():
                        partition_key = f"{partition_value}"
                        
                        if partition_key not in results:
                            results[partition_key] = {
                                "densities": [],
                                "flows": [],
                                "flow_stds": [],
                                "file_names": []
                            }
                        
                        # Filter data for this partition value
                        filtered_df = df[df[partition_col] == partition_value]
                        
                        # Sort by density for proper plotting
                        filtered_df = filtered_df.sort_values(by="Density")
                        
                        # Add to results
                        results[partition_key]["densities"].append(filtered_df["Density"].values)
                        results[partition_key]["flows"].append(filtered_df["Average Flow"].values)
                        
                        # Add standard deviation if available
                        if "Standard Deviation of Flow" in filtered_df.columns:
                            results[partition_key]["flow_stds"].append(filtered_df["Standard Deviation of Flow"].values)
                        else:
                            # Use zeros if standard deviation not available
                            results[partition_key]["flow_stds"].append(np.zeros_like(filtered_df["Average Flow"].values))
                        
                        results[partition_key]["file_names"].append(file_name)
                        
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
    
    return results


def create_visualization(results, folder_path, partition_type="distraction"):
    """
    Create a visualization of flow vs density with error bars and splines.
    
    Args:
        results (dict): Analysis results from analyze_excel_files
        folder_path (str): Path to the folder where the output image will be saved
        partition_type (str): Type of partition used ('distraction' or 'distribution')
    """
    plt.figure(figsize=(12, 8))
    
    # Sort partition keys by value and convert to numeric if possible
    partition_keys = list(results.keys())
    try:
        # Try to convert keys to numeric for proper sorting
        partition_keys.sort(key=float)
    except:
        # Fall back to regular sorting if conversion fails
        partition_keys.sort()
    
    # Define color map for different partition values
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    
    # Plot each partition value with a different color in sorted order
    for i, partition_key in enumerate(partition_keys):
        data = results[partition_key]
        color = colors[i]
        # Combine data from all files with the same partition value
        all_densities = np.concatenate(data["densities"])
        all_flows = np.concatenate(data["flows"])
        all_flow_stds = np.concatenate(data["flow_stds"])
        
        # Group and compute statistics for points with the same density
        unique_densities = np.sort(np.unique(all_densities))
        avg_flows = []
        flow_std_errs = []
        
        for density in unique_densities:
            density_mask = (all_densities == density)
            density_flows = all_flows[density_mask]
            density_flow_stds = all_flow_stds[density_mask]
            
            # Calculate average flow for this density
            avg_flow = np.mean(density_flows)
            avg_flows.append(avg_flow)
            
            # For standard error, use the average of individual standard deviations
            # If the std devs are zeros (not available), calculate it from the flow values
            if np.all(density_flow_stds == 0) and len(density_flows) > 1:
                flow_std_err = np.std(density_flows, ddof=1) / np.sqrt(len(density_flows))
            else:
                flow_std_err = np.mean(density_flow_stds) / np.sqrt(len(density_flows))
            
            flow_std_errs.append(flow_std_err)
        
        # Create a linear spline interpolation between points
        if len(unique_densities) > 1:
            spline = interp1d(unique_densities, avg_flows, kind='linear')
            x_dense = np.linspace(min(unique_densities), max(unique_densities), 100)
            y_dense = spline(x_dense)
            
            # Plot spline
            plt.plot(x_dense, y_dense, color=color, label=f"{partition_type.capitalize()} = {partition_key}%")
        else:
            # If there's only one point, we can't create a spline
            plt.plot(unique_densities, avg_flows, 'o-', color=color, 
                    label=f"{partition_type.capitalize()} = {partition_key}%")
        
        # Plot error bars
        plt.errorbar(unique_densities, avg_flows, yerr=flow_std_errs, fmt='o', color=color, capsize=5)
    
    # Set labels and title
    plt.xlabel('Density (cars/km/lane)')
    plt.ylabel('Flow (cars/hour)')
    plt.title('Flow vs Density by ' + partition_type.capitalize() + " Percentage")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the figure in the input folder path
    output_path = os.path.join(folder_path, f'flow_vs_density_{partition_type}.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Image saved to: {output_path}")
    plt.show()


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Traffic Flow Analysis Tool')
    parser.add_argument('folder_path', type=str, help='Path to folder containing Excel files')
    parser.add_argument('--partition', type=str, choices=['distraction', 'distribution'], 
                        default='distraction', help='Type of partition to use')
    parser.add_argument('--output', type=str, help='Custom output filename (optional)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Analyze files
    results = analyze_excel_files(args.folder_path, args.partition)
    
    # Create visualization
    if results:
        # Pass the folder path to create_visualization function
        create_visualization(results, args.folder_path, args.partition)
        print("Analysis complete.")
    else:
        print("No valid data found for analysis.")


if __name__ == "__main__":
    main()