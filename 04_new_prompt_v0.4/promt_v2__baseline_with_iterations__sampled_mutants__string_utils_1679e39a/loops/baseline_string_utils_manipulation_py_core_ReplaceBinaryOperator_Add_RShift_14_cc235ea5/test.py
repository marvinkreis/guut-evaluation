from string_utils.manipulation import camel_case_to_snake

def test__camel_case_to_snake():
    """
    Test whether the conversion from camel case to snake case is performed correctly.
    The input 'ThisIsACamelStringTest' should translate to 'this_is_a_camel_string_test'.
    The mutant changes the behavior of the conversion by mistakenly using the bitwise right shift operator '>>'
    instead of the concatenation operator, which will produce an incorrect result.
    """
    output = camel_case_to_snake('ThisIsACamelStringTest')
    assert output == 'this_is_a_camel_string_test'