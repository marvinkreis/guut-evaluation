from string_utils.validation import is_json

def test__is_json():
    """The mutant fails when processing malformed JSON due to NameError on undefined CosmicRayTestingException."""
    malformed_json = '{nope}'  # This should trigger an exception in the mutant
    correct_output = is_json(malformed_json)
    
    assert correct_output is False, "The correct code should return False for malformed JSON."