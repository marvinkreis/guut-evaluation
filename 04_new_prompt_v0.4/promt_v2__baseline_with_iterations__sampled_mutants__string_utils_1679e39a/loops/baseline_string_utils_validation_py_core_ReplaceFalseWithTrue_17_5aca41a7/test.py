from string_utils.validation import is_pangram

def test__is_pangram():
    """
    Test whether the function correctly identifies a string containing all letters of the 
    English alphabet as a pangram. The input is a valid pangram, which should return 
    True in the baseline, but the mutant will incorrectly return True also for 
    non-pangram inputs (like empty strings) and might fail to check the content.
    """
    output = is_pangram('')
    assert output == False