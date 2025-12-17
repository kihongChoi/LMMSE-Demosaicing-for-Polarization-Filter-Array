"""Quick test for blockproc function."""

import numpy as np
from Function.blockproc import blockproc


def test_blockproc_basic():
    """Test basic blockproc functionality."""
    print("Testing blockproc...")
    
    # Create test image
    img = np.random.rand(100, 100).astype(np.float64)
    
    # Simple function that doubles the values
    def fun(x):
        return x * 2
    
    # Test without border
    result = blockproc(img, (10, 10), fun, border_size=(0, 0), trim_border=True)
    print(f"  Input shape: {img.shape}")
    print(f"  Output shape: {result.shape}")
    print(f"  Test passed: {result.shape == img.shape}")
    
    # Test with border
    result2 = blockproc(img, (10, 10), fun, border_size=(2, 2), trim_border=True)
    print(f"  With border - Output shape: {result2.shape}")
    print(f"  Test passed: {result2.shape == img.shape}")
    
    print("blockproc test completed!\n")


if __name__ == '__main__':
    test_blockproc_basic()
