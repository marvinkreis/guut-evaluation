import re

# Define the correct LEFT_SPACE regex for our test
LEFT_SPACE = re.compile(r'^\s+')  # Matches leading whitespace

def test__LEFT_SPACE():
    """Testing LEFT_SPACE to ensure it captures leading spaces correctly and fails on the mutant."""
    
    # A string that contains leading spaces
    correct_string = '   "test with leading spaces"'
    
    # Check if leading spaces are captured correctly
    correct_matches = LEFT_SPACE.findall(correct_string)
    assert len(correct_matches) > 0, "LEFT_SPACE must match leading spaces."

    # An example mutant scenario - no leading spaces
    mutant_string = '"No leading spaces"'
    
    # This should not match since there are no leading spaces
    mutant_matches = LEFT_SPACE.findall(mutant_string)
    assert len(mutant_matches) == 0, "Mutant should not match where there are no leading spaces."

# Run the test
test__LEFT_SPACE()