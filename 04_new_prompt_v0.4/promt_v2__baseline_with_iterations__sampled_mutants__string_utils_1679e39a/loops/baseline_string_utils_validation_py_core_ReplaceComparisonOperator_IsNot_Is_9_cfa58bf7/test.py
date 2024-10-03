from string_utils.validation import is_uuid

def test__is_uuid():
    """
    Test whether the function correctly identifies a valid UUID when allow_hex is True.
    The input '6f8aa2f9-686c-4ac3-8766-5712354a04cf' is a valid UUID that should return True.
    The mutant incorrectly matches the UUID by returning False instead of True, thus failing.
    """
    output = is_uuid('6f8aa2f9-686c-4ac3-8766-5712354a04cf', allow_hex=True)
    assert output == True