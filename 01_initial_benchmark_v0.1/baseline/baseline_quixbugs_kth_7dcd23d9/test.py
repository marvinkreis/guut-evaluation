from kth import kth

def test__kth():
    arr = [3, 1, 7, 5, 4, 2, 6]  # Unsorted array of integers
    k_original = 3  # We want the 4th-lowest element (0-based index), which should be 4.
    
    result_correct = kth(arr, k_original)

    # The original function should return 4, which is the 4th lowest
    assert result_correct == 4, f"Expected {4}, but got {result_correct}"

    # Now validate the mutant behavior
    def mutant_kth(arr, k):
        if not arr:  # Safeguard against empty list calls
            return None
        
        pivot = arr[0]
        below = [x for x in arr if x < pivot]
        above = [x for x in arr if x > pivot]

        num_less = len(below)
        num_lessoreq = len(arr) - len(above)

        if k < num_less:
            return mutant_kth(below, k)
        elif k >= num_lessoreq:
            return mutant_kth(above, k)  # This line reflects the mutation
        else:
            return pivot

    # Testing the mutant with the same k
    result_mutant = mutant_kth(arr, k_original)

    # Check that the mutant result is different since we expect errors in logic
    assert result_mutant != 4, f"Mutant should produce a different result than {4}, but got {result_mutant}"