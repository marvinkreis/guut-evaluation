import re

# Create a small function to test the regex pattern directly
def test_regex_for_duplicates():
    # This is the regex we want to check against
    pattern = r'(\!{2,})'  # This specifically matches two or more exclamation marks
    test_string = "Wow!!! Amazing!!!"
    
    # Testing the regex directly
    matches = re.findall(pattern, test_string)
    print("Direct regex matches for '!!!':", matches)
    assert len(matches) > 0, "Should find duplicates in '!!!' through direct regex testing"

# Call the testing function to check the behavior
test_regex_for_duplicates()