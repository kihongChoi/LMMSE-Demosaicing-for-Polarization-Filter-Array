"""Quick test for load_dataset function."""

import os
from Function.load_dataset import load_dataset


def test_load_dataset():
    """Test load_dataset functionality."""
    print("Testing load_dataset...")
    
    folder_path = 'Data/Dataset'
    
    if not os.path.exists(folder_path):
        print(f"  ERROR: Folder {folder_path} does not exist!")
        return
    
    try:
        dataset = load_dataset(
            save=False,
            folder=folder_path,
            nbr_of_img=1,
            mosaic='pfa'
        )
        
        print(f"  Dataset loaded: {len(dataset)} images")
        if len(dataset) > 0:
            name, img_0, img_45, img_90, img_135 = dataset[0]
            print(f"  Image name: {name}")
            print(f"  Image 0° shape: {img_0.shape}, dtype: {img_0.dtype}")
            print(f"  Image 45° shape: {img_45.shape}, dtype: {img_45.dtype}")
            print(f"  Value range: [{img_0.min():.3f}, {img_0.max():.3f}]")
            print("  Test passed!")
        else:
            print("  WARNING: No images loaded")
            
    except Exception as e:
        print(f"  ERROR: {e}")
    
    print("load_dataset test completed!\n")


if __name__ == '__main__':
    test_load_dataset()
