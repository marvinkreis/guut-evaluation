from string_utils.validation import is_isbn_10

def test__is_isbn_10_mutant_killer():
    """
    Test valid ISBN 10 number '1506715214'. The original function should return True, while the mutant version will return False
    due to the incorrect weight calculation in the mutated code.
    """
    isbn_valid = '1506715214'
    result = is_isbn_10(isbn_valid)
    assert result == True, f"Expected True, got {result}"

test__is_isbn_10_mutant_killer()