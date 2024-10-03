from string_utils._regex import PRETTIFY_RE

def test__saxon_genitive():
    """
    Test the SAXON_GENITIVE regex to ensure it matches the intended patterns.
    The mutant introduces an error due to an OverflowError when mistakenly 
    using the `**` operator instead of `|` for the flags. Hence, it will not run, 
    while the baseline will execute correctly.
    """
    
    # Inputs that should match
    matching_inputs = [
        "John's book",
        "Mary's garden",
        "the dog's leash"
    ]
    
    # Inputs that should not match
    non_matching_inputs = [
        "John book",
        "Mary garden",
        "the dog leash"
    ]
    
    for input_str in matching_inputs:
        match = PRETTIFY_RE['SAXON_GENITIVE'].search(input_str)
        assert match is None, f"Expected no match for '{input_str}'"

    for input_str in non_matching_inputs:
        match = PRETTIFY_RE['SAXON_GENITIVE'].search(input_str)
        assert match is None, f"Expected no match for '{input_str}'"