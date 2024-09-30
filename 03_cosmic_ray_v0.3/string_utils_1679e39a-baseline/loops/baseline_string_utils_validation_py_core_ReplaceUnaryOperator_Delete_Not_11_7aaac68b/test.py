from string_utils.validation import words_count

def test__words_count():
    """
    Test whether the function correctly counts the number of words in a string.
    The input 'hello world' should return 2 words, but if the mutant is applied,
    it will incorrectly raise an InvalidInputError due to the faulty check for 
    input type being the opposite condition (i.e., it checks if it is a string).
    """
    output = words_count('hello world')
    assert output == 2