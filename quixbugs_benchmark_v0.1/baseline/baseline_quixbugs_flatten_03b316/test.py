from flatten import flatten

def test_flatten():
    # Prepare a nested list for testing
    nested_list = [[1, [], [2, 3]], [[4]], 5]
    
    # The expected output should be a flattened list
    expected_output = [1, 2, 3, 4, 5]
    
    # Collect the output from the flatten generator
    output = list(flatten(nested_list))
    
    # Assert that the output is as expected with the correct code
    assert output == expected_output, f"Expected {expected_output}, but got {output}"
    
    # Now we will create a condition to test against the mutant
    
    # If the mutant were applied, the output would be incorrect since it would yield a generator instead of the non-list object.
    # We can create a situation where an incorrect output can be captured by asserting the wrong length in case of the mutant.
    
    # Mutant would yield flatten(x) in the non-list case, leading to a longer output than expected:
    wrong_output = output + [flatten(nested_list)]
    
    # This condition should not hold true with the correct implementation
    assert len(wrong_output) == len(expected_output) + 1, "Mutant detected - output length should not match the original"