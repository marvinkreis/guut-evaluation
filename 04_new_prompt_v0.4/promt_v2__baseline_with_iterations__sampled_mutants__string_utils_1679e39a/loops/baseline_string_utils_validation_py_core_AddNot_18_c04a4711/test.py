from string_utils.validation import is_uuid

def test__is_uuid():
    """
    Test whether the function accurately validates the UUID format when the allow_hex flag is set to False. 
    A valid UUID in standard format should be recognized, but if allow_hex is False, a hex format should not pass.
    The input '6f8aa2f9-686c-4ac3-8766-5712354a04cf' is a valid UUID and should return True, while the 
    input '6f8aa2f9686c4ac387665712354a04cf' is a valid hex representation that should return False if allow_hex is False.
    """
    output_standard = is_uuid('6f8aa2f9-686c-4ac3-8766-5712354a04cf', allow_hex=False)
    assert output_standard == True
    
    output_hex = is_uuid('6f8aa2f9686c4ac387665712354a04cf', allow_hex=False)
    assert output_hex == False