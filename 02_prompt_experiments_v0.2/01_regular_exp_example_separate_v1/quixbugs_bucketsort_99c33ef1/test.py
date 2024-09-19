from bucketsort import bucketsort

def test__bucketsort():
    """
    Changing the enumeration in the loops of bucketsort leads to incorrect sorting behavior.
    This test checks that correct inputs indeed produce a correct and properly sorted output,
    while the mutant fails to replicate this behavior.
    """
    input_array = [3, 1, 4, 1, 5, 9]  # Unsorted input
    upper_bound = 10                  # Upper bound for integers in the input

    # Testing correct functionality
    correct_output = bucketsort(input_array, upper_bound)

    # Testing the mutant directly
    def mutant_bucketsort(arr, k):
        counts = [0] * k
        for x in arr:
            counts[x] += 1
    
        sorted_arr = []
        for i, count in enumerate(arr):  # This is the mutant line
            sorted_arr.extend([i] * count)
    
        return sorted_arr
      
    mutant_output = mutant_bucketsort(input_array, upper_bound)
    
    # Assertions to check outputs
    assert correct_output == [1, 1, 3, 4, 5, 9], "Correct bucketsort should properly sort the array."
    assert mutant_output != correct_output, "Mutant bucketsort should not match correct output."