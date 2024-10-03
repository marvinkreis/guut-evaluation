from string_utils.validation import is_isbn_13

def test__is_isbn_13():
    """
    Test whether the function is_isbn_13 correctly handles ISBN with hyphens. The input '978-0312498580' is a valid 
    ISBN 13 number; it should return True under the baseline and False under the mutant due to the changed 
    default for the normalize parameter.
    """
    output = is_isbn_13('978-0312498580')
    assert output == True, f"Expected True, but got {output}."