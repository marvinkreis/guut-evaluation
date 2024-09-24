from string_utils.validation import is_isogram

def test__is_isogram():
    # Testing a valid isogram
    isogram_input = "dermatoglyphics"
    assert is_isogram(isogram_input) == True, "The string should be recognized as an isogram"

    # Testing a non-isogram
    non_isogram_input = "hello"
    assert is_isogram(non_isogram_input) == False, "The string should not be recognized as an isogram"