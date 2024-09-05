from knapsack import knapsack

def test_knapsack():
    capacity = 100
    items = [(60, 10), (50, 8), (20, 4), (20, 4), (8, 3), (3, 2)]
    expected_value = 19  # The expected result given in the problem statement

    # Test the original function
    result = knapsack(capacity, items)
    
    # Check if the result matches the expected value
    assert result == expected_value, f"Expected {expected_value}, but got {result}"

    # Create another test case that will expose the mutant
    # Here we are adding an item that has weight equal to the remaining capacity
    additional_item = [(100, 50)]  # Adding an item that exactly matches the capacity
    result_with_extra = knapsack(capacity, items + additional_item)

    # In the original code, the maximum value should be 50 from the additional item
    assert result_with_extra == 50, f"Expected 50, but got {result_with_extra}"

    # If the mutant code is executed, it would return something different than 50,
    # thus by checking against both original conditions we can detect the mutant.
