from string_utils.validation import is_isbn_10

def test__is_isbn_10():
    # Test case 1: Valid ISBN-10
    valid_isbn_10 = '1506715214'
    assert is_isbn_10(valid_isbn_10) == True, f"Expected True for valid ISBN-10: {valid_isbn_10}"

    # Test case 2: Valid ISBN-10 with hyphens
    valid_isbn_10_with_hyphens = '150-6715214'
    assert is_isbn_10(valid_isbn_10_with_hyphens) == True, f"Expected True for valid ISBN-10 with hyphens: {valid_isbn_10_with_hyphens}"

    # Test case 3: Invalid ISBN-10
    invalid_isbn_10 = '1506715215'
    assert is_isbn_10(invalid_isbn_10) == False, f"Expected False for invalid ISBN-10: {invalid_isbn_10}"

    # Test case 4: Valid ISBN-10 with normalization disabled
    assert is_isbn_10(valid_isbn_10_with_hyphens, normalize=False) == False, \
        f"Expected False for valid ISBN-10 with normalization disabled: {valid_isbn_10_with_hyphens}"

# Running the test
# If this is part of a script, uncomment the line below
# test__is_isbn_10()