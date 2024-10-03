from kth import kth

def test__kth_final_check():
    """
    Verify that the kth function behaves as expected on edge cases.
    The baseline implementation should return 1 for k=0 and 5 for k=4,
    while the mutant will raise an IndexError due to incorrect logic in handling k.
    This test confirms the disparate behavior between the two implementations.
    """
    arr = [3, 1, 2, 5, 4]
    
    # For k=0, expect the smallest element
    assert kth(arr, 0) == 1, "Expected 1 for k=0"

    # For k=4, expect the largest element, should fail with mutant
    assert kth(arr, 4) == 5, "Expected 5 for k=4"