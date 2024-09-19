from powerset import powerset

def test_powerset():
    # Test case to check the power set
    input_data = ['a', 'b', 'c']
    expected_output = [[], ['c'], ['b'], ['b', 'c'], ['a'], ['a', 'c'], ['a', 'b'], ['a', 'b', 'c']]
    
    # Call the powerset function with the input data
    result = powerset(input_data)
    
    # Assert the result matches the expected output
    assert result == expected_output, f"Expected {expected_output} but got {result}"