"""Quick test for function_lmmse_demosaicing function."""

import os
import numpy as np
import scipy.io as sio
from PIL import Image
from function_lmmse_demosaicing import function_lmmse_demosaicing


def test_demosaicing():
    """Test function_lmmse_demosaicing functionality."""
    print("Testing function_lmmse_demosaicing...")
    
    # Check if D matrix exists
    d_matrix_path = 'Data/D_Matrix.mat'
    if not os.path.exists(d_matrix_path):
        print(f"  ERROR: D matrix file {d_matrix_path} does not exist!")
        print("  Please ensure D_Matrix.mat is in the Data folder.")
        return
    
    # Check if mosaiced image exists
    mos_img_path = 'Data/im.tif'
    if not os.path.exists(mos_img_path):
        print(f"  ERROR: Mosaiced image {mos_img_path} does not exist!")
        print("  Please ensure im.tif is in the Data folder.")
        return
    
    try:
        # Load D matrix
        print("  Loading D matrix...")
        d_matrix_data = sio.loadmat(d_matrix_path)
        D = d_matrix_data['D']
        print(f"  D matrix shape: {D.shape}")
        
        # Load mosaiced image
        print("  Loading mosaiced image...")
        mos_img = np.array(Image.open(mos_img_path), dtype=np.float64) / 255.0
        
        # Handle grayscale
        if len(mos_img.shape) == 3 and mos_img.shape[2] == 1:
            mos_img = mos_img[:, :, 0]
        elif len(mos_img.shape) == 3:
            mos_img = np.mean(mos_img, axis=2)  # Convert RGB to grayscale
        
        print(f"  Mosaiced image shape: {mos_img.shape}")
        
        # Demosaicing
        print("  Running demosaicing...")
        demos_img = function_lmmse_demosaicing(mos_img, D)
        
        print(f"  Demosaiced image shape: {demos_img.shape}")
        print(f"  Expected shape: ({mos_img.shape[0]}, {mos_img.shape[1]}, 1, 4)")
        print(f"  Value range: [{demos_img.min():.3f}, {demos_img.max():.3f}]")
        print("  Test passed!")
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print("function_lmmse_demosaicing test completed!\n")


if __name__ == '__main__':
    test_demosaicing()
