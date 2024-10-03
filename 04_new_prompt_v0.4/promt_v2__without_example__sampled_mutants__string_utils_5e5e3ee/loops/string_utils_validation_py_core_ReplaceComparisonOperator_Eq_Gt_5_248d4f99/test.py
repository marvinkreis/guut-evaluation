from string_utils.validation import is_isogram

def test__mutant_killing_is_isogram():
    """
    This test ensures that the mutant is detected. 
    The input 'abc' is an isogram and should return True 
    on the baseline while returning False on the mutant 
    due to the erroneous greater-than condition.
    """
    unique_char_input = 'abc'
    
    # Running the check to see if it identifies correctly
    assert is_isogram(unique_char_input) == True