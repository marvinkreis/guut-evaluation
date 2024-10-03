from string_utils.validation import is_json

def test__is_json():
    """
    Test the is_json function to ensure valid JSON strings return True. The mutant is expected to fail
    and return False due to its altered logic.
    """
    valid_json_inputs = [
        '{"name": "Peter"}', 
        '[1, 2, 3]', 
        '{"key": null, "is_valid": true}',
        '["apple", "banana", "cherry"]'
    ]
    
    for json_input in valid_json_inputs:
        output = is_json(json_input)
        assert output == True, f"Expected True for input: {json_input}, but got {output}"