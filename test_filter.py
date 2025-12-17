"""Quick test for filter_images function."""

import numpy as np
from Function.filter import filter_images


def test_filter_images():
    """Test filter_images functionality."""
    print("Testing filter_images...")
    
    # Create test image with 4 channels (0°, 45°, 90°, 135°)
    rows, cols = 100, 100
    test_img = np.random.rand(rows, cols, 4).astype(np.float64)
    
    try:
        filtered = filter_images(test_img, mosaic='pfa')
        print(f"  Input shape: {test_img.shape}")
        print(f"  Output shape: {filtered.shape}")
        print(f"  Output dtype: {filtered.dtype}")
        print(f"  Value range: [{filtered.min():.3f}, {filtered.max():.3f}]")
        print("  Test passed!")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print("filter_images test completed!\n")


if __name__ == '__main__':
    test_filter_images()
