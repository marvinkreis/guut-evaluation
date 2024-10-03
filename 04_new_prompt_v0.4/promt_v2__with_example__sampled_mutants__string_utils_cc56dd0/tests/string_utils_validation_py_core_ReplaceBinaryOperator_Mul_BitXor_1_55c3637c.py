from string_utils.validation import is_isbn_10

def test_is_isbn_10_mutant_killing():
    """
    This test checks the is_isbn_10 function with a known valid ISBN-10 number.
    The mutant's change to use XOR instead of multiplication will cause it to fail,
    resulting in a False output for valid ISBN-10 numbers.
    """
    output = is_isbn_10('0471958697')
    assert output == True, f"Expected True, got {output}"