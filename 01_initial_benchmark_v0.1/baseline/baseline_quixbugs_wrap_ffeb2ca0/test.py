from wrap import wrap

def test_wrap():
    text = "This is a long sentence that we will use to test the wrapping logic."
    cols = 20
    expected_output = [
        "This is a long",
        "sentence that we",
        "will use to test",
        "the wrapping logic."
    ]
    
    result = wrap(text, cols)
    
    # Debugging output for clarity
    print("Result:", result)
    print("Expected:", expected_output)

    # Verify that the output matches the expected output after stripping leading spaces
    assert [line.strip() for line in result] == expected_output, f"Expected {expected_output}, but got {[line.strip() for line in result]}"

    # Ensure that the result contains the correct number of lines
    assert len(result) == 4, "The mutant may have omitted a line."