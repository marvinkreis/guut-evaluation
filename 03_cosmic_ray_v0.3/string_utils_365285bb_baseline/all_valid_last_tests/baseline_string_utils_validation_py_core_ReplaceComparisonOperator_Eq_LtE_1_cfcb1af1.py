from string_utils.validation import is_isbn_10

def test__is_isbn_10():
    # A known valid ISBN-10
    valid_isbn_1 = '0471958697'  # This is a valid ISBN-10
    assert is_isbn_10(valid_isbn_1) == True, "Expected True for valid ISBN-10"

    # Change one digit to invalidate
    invalid_isbn_1 = '0471958693'  # Modify last digit to make it invalid
    assert is_isbn_10(invalid_isbn_1) == False, "Expected False for altered ISBN-10"

    # Another valid ISBN-10
    valid_isbn_2 = '0306406152'  # This is valid
    assert is_isbn_10(valid_isbn_2) == True, "Expected True for another valid ISBN-10"

    # An ISBN with an altered digit aiming to expose the mutant logic issue
    mutant_test_case = '0306406151'  # This is intended to fail 
    assert is_isbn_10(mutant_test_case) == False, "Expected False due to mutant logic on weight calculation"

    # Another clearly invalid ISBN-10
    clearly_invalid_isbn = '1234567890'  # Invalid due to structure
    assert is_isbn_10(clearly_invalid_isbn) == False, "Expected False for clearly invalid ISBN-10"

    # Additional test for edge case
    edge_case_valid_isbn = '2800103000'  # Valid ISBN
    assert is_isbn_10(edge_case_valid_isbn) == True, "Expected True for an edge case valid ISBN-10"

    # Altering a valid edge case to be invalid
    edge_case_invalid_isbn = '2800103001'  # Change to make it invalid
    assert is_isbn_10(edge_case_invalid_isbn) == False, "Expected False for altered edge case ISBN-10"
