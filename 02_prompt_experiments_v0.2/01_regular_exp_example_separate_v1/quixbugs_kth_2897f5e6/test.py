from kth import kth  # Ensure the correct function is imported

def kth_mutant(arr, k):
    if not arr:
        raise ValueError("Empty array is not valid")
        
    pivot = arr[0]
    below = [x for x in arr if x < pivot]
    above = [x for x in arr if x > pivot]

    num_less = len(below)
    num_lessoreq = len(arr) - len(above)

    if k < num_less:
        return kth(below, k)
    elif k >= num_lessoreq:
        return kth(above, k)  # Mutant change
    else:
        return pivot

def test__kth():
    """Test to kill the mutant. The mutant returns an incorrect kth value."""
    arr = [1, 2, 3, 4, 5]
    k = 2
    output = kth(arr, k)
    assert output == 3, "The correct implementation must return 3"
    
    output_mutant = kth_mutant(arr, k)
    assert output_mutant == 4, "The mutant implementation returns incorrect output"

# Call the test function to execute the test
test__kth()