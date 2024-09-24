from string_utils.validation import contains_html

def test__contains_html():
    """Testing contains_html function to ensure it handles inputs correctly.
    The mutant fails on valid string inputs and provides error messages on invalid ones.
    """
    assert contains_html('<div>Hello</div>') == True, "Should detect HTML"
    assert contains_html('Hello') == False, "Should not detect HTML"
    assert contains_html('<p>Paragraph</p>') == True, "Should detect HTML"
    assert contains_html('No HTML here') == False, "Should not detect HTML"
    
    try:
        contains_html(None)
        assert False, "Should raise InvalidInputError for None"
    except Exception as e:
        assert str(e) == 'Expected "str", received "NoneType"', "Incorrect error message for None"

    try:
        contains_html(123)
        assert False, "Should raise InvalidInputError for non-string"
    except Exception as e:
        assert str(e) == 'Expected "str", received "int"', "Incorrect error message for non-string"
