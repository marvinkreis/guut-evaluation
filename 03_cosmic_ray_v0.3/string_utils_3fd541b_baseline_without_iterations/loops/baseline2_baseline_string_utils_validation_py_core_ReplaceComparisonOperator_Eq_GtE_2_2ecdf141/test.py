from string_utils.validation import is_isbn_13

def test__is_isbn_13():
    # Valid ISBN-13 example
    assert is_isbn_13('9780312498580') == True
    # Invalid ISBN-13 example (correct calculation: 9780312498580 has a check digit of 0, valid)
    assert is_isbn_13('9780312498581') == False
    # Further test case for invalid ISBN
    assert is_isbn_13('1234567890123') == False  # Random invalid number

# Note: Execute the test function to verify it passes in the correct code and fails with the mutant.