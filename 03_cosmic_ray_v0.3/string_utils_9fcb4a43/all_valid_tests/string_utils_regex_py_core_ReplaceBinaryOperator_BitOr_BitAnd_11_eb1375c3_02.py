import re
from string_utils._regex import SPACES_RE

# Redefining the mutant regex to reflect the mutation
# Changing | to & in the regex pattern we're testing
mutant_SPACES_RE = re.compile(r'(?<=\S)\+(?=\S)|(?<=\S)\+\s|\s\+(?=\S)')

def test__spaces_regex():
    """The mutant version of the REGEX for SPACES_RE should not function correctly due to the change from | to &."""
    
    # Test strings that should produce leading spaces output
    test_inputs = [
        " A sentence with a leading space.",   # Expected to match
        "Somesentencewithoutleadingortrailing." # Expected not to match
    ]
    
    for input_str in test_inputs:
        print(f"Testing input: '{input_str}'")
        
        # Execute correct implementation
        correct_matches = SPACES_RE.findall(input_str)
        print(f"Correct matches: {correct_matches}")
        
        # Execute mutant behavior
        mutant_matches = mutant_SPACES_RE.findall(input_str)
        print(f"Mutant matches: {mutant_matches}")
        
        # Ensure the test conditions
        if " A sentence" in input_str:
            assert len(correct_matches) > 0, "The correct SPACES_RE should find leading spaces."
            assert len(mutant_matches) == 0, "The mutant SPACES_RE should produce an empty output for leading spaces."
        else:
            assert len(mutant_matches) == 0, "Expect no matches in mutant for this input."

# Execute the test
test__spaces_regex()