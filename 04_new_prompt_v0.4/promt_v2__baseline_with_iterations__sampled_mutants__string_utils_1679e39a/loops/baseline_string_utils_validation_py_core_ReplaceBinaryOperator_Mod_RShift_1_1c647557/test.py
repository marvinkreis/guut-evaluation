from string_utils.validation import is_isbn_13

def test__is_isbn_13():
    """
    Test whether an invalid ISBN-13 number is correctly identified. 
    This test uses '978-0-306-40615-8', which is an invalid ISBN-13. 
    The mutant changes '%' to '>>' in the validation check, which will prevent
    the correct failure for this invalid input, leading to it returning True instead of False.
    """
    output = is_isbn_13('978-0-306-40615-8')
    assert output == False