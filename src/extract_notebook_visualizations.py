#!/usr/bin/env python3
"""
Extract and save visualizations from Jupyter notebooks.

This script runs through a Jupyter notebook and saves all matplotlib figures to disk.
"""
import os
import sys
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import matplotlib.pyplot as plt
import io
from PIL import Image
import base64

# Add parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def save_notebook_visualizations(notebook_path, output_dir):
    """
    Execute a notebook and save all visualizations to the output directory.
    
    Args:
        notebook_path: Path to the Jupyter notebook
        output_dir: Directory to save visualizations
    
    Returns:
        List of paths to saved visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the notebook filename without extension
    notebook_name = os.path.splitext(os.path.basename(notebook_path))[0]
    
    # Load the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)
    
    # Configure the execution environment
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    
    # Save current matplotlib backend
    original_backend = plt.get_backend()
    
    # Use Agg backend (non-interactive)
    plt.switch_backend('Agg')
    
    # Execute the notebook
    print(f"Executing notebook: {notebook_path}")
    try:
        ep.preprocess(notebook, {'metadata': {'path': os.path.dirname(notebook_path)}})
    except Exception as e:
        print(f"Error executing notebook: {e}")
        plt.switch_backend(original_backend)
        return []
    
    # List to store paths of saved visualizations
    saved_files = []
    
    # Extract and save figures from outputs
    print("Extracting visualizations...")
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == 'code' and 'outputs' in cell:
            for j, output in enumerate(cell.outputs):
                # For png/jpeg data
                if 'data' in output and ('image/png' in output.data or 'image/jpeg' in output.data):
                    img_data = None
                    img_format = None
                    
                    if 'image/png' in output.data:
                        img_data = output.data['image/png']
                        img_format = 'png'
                    elif 'image/jpeg' in output.data:
                        img_data = output.data['image/jpeg']
                        img_format = 'jpg'
                    
                    if img_data:
                        # Decode base64 data
                        img_bytes = base64.b64decode(img_data)
                        
                        # Open the image using PIL
                        img = Image.open(io.BytesIO(img_bytes))
                        
                        # Save the image
                        file_path = os.path.join(output_dir, f"{notebook_name}_cell{i}_output{j}.{img_format}")
                        img.save(file_path)
                        
                        saved_files.append(file_path)
                        print(f"Saved visualization to: {file_path}")
    
    # Restore original backend
    plt.switch_backend(original_backend)
    
    return saved_files

def main():
    # Define paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    notebook_path = os.path.join(base_dir, 'notebooks', 'Metrics.ipynb')
    output_dir = os.path.join(base_dir, 'results', 'notebook_visualizations')
    
    # Save visualizations from the notebook
    saved_files = save_notebook_visualizations(notebook_path, output_dir)
    
    if saved_files:
        print(f"\nSuccessfully saved {len(saved_files)} visualizations to {output_dir}")
    else:
        print("\nNo visualizations were saved.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
