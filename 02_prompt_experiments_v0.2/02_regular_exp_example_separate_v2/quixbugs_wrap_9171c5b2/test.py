from wrap import wrap  # Ensure this import is correct

def test__wrap():
    """The mutant loses the last segment of the string due to the omission of the last append operation."""
    text = "This is an example string that needs to be tested"
    cols = 22
    output = wrap(text, cols)  # Invoke the function under test
    expected_output = ['This is an example', ' string that needs to', ' be tested']
    assert output == expected_output, f"Expected {expected_output}, but got {output}"

# Run the test
test__wrap()