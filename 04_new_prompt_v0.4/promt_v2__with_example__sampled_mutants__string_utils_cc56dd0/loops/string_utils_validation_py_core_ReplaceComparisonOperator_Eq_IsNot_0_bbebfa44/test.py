from string_utils.validation import is_isogram

def test_is_isogram_mutant_killing():
    """
    Test the is_isogram function using a unique character string and a non-unique character string.
    The baseline should return True for the unique string and False for the non-unique string,
    while the mutant will incorrectly return False for the unique string.
    """
    # Test with a unique character string
    unique_output = is_isogram("abcdefg")
    assert unique_output == True, f"Expected True, got {unique_output}"
    
    # Test with a non-unique character string
    non_unique_output = is_isogram("hello")
    assert non_unique_output == False, f"Expected False, got {non_unique_output}"