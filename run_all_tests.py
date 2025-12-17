"""Run all tests in order."""

import sys

def run_test(module_name):
    """Run a test module."""
    print("=" * 60)
    print(f"Running {module_name}")
    print("=" * 60)
    try:
        __import__(module_name)
    except Exception as e:
        print(f"ERROR in {module_name}: {e}")
        import traceback
        traceback.print_exc()
        return False
    return True


def main():
    """Run all tests in order."""
    tests = [
        'test_blockproc',
        'test_load_dataset',
        'test_filter',
        'test_demosaicing',
    ]
    
    print("\n" + "=" * 60)
    print("LMMSE Demosaicing - Test Suite")
    print("=" * 60 + "\n")
    
    results = []
    for test in tests:
        success = run_test(test)
        results.append((test, success))
        print()
    
    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"  {test}: {status}")
    
    all_passed = all(success for _, success in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED. Please check the errors above.")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
