from string_utils.validation import is_isbn

def test__is_isbn_killing_mutant():
    """
    This test checks that the validity of ISBNs works correctly. 
    The input '1506715214' is a valid ISBN-10, which should pass the is_isbn function 
    in the baseline, but fail in the mutant since the mutant requires both ISBN-10 
    and ISBN-13 validation to be True.
    """
    input_string = '1506715214'
    output = is_isbn(input_string)
    assert output == True