from string_utils.manipulation import prettify

def test__prettify():
    # Test input with multiple placeholders that should trigger the replacement logic
    input_string = ' Test text with placeholder $fake_placeholder$ and another one $fake_placeholder$   '
    
    # The expected output should properly restore the placeholder once
    # If the mutant is in place, it will replace the placeholder twice which is incorrect
    expected_output = 'Test text with placeholder $fake_placeholder$ and another one $fake_placeholder$'
    
    # Execute the function and assert
    result = prettify(input_string)
    assert result == expected_output, f"Expected: {expected_output}, but got: {result}"

# We can execute the test case to check against the correct version and the mutant
test__prettify()