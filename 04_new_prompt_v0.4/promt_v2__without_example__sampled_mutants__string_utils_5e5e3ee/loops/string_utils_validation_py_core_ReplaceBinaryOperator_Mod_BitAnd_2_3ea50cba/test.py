from string_utils.validation import is_isbn_10

def test__kill_mutant_isbn():
    """
    Test the behavior of is_isbn_10 with the valid ISBN '0306406152'.
    The expected behavior is that it should return True for the baseline due to being a valid ISBN-10.
    The mutant is expected to return False due to its altered logic via the AND operation instead of modulo.
    """
    isbn = '0306406152'  # A valid ISBN-10
    
    output = is_isbn_10(isbn)
    print(f"output for '0306406152': {output}")

    assert output == True  # Expecting it to validate successfully in the baseline but likely fail in the mutant.