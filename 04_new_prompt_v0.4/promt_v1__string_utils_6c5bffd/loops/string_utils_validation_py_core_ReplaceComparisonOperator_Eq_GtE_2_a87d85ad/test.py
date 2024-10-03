from string_utils.validation import is_isbn_13

def test__is_isbn_13_mutation_killer():
    """
    Test whether the is_isbn_13 function correctly validates ISBN-13 numbers. 
    A valid ISBN-13 ('9780312498580') should return True, and an 
    invalid ISBN-13 ('9780312498581') should return False. The mutant is expected 
    to return True for the invalid ISBN, so this test will fail on the mutant.
    """
    valid_isbn = '9780312498580'  # Valid ISBN-13
    invalid_isbn = '9780312498581'  # Invalid ISBN-13
    
    assert is_isbn_13(valid_isbn) is True
    assert is_isbn_13(invalid_isbn) is False