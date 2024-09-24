from string_utils._regex import PRETTIFY_RE

def test_PRETTIFY_RE():
    # Test input containing several duplicate spaces
    input_string = "Hello    World!   This is   a test!    "
    
    # Expected to capture
    # 1. "    " between "Hello" and "World!"
    # 2. "   " between "World!" and "This"
    # 3. "   " between "is" and "a"
    # 4. "    " after "test!"
    expected_matches_count = 4  # Total of 4 matches expected

    matches = PRETTIFY_RE['DUPLICATES'].findall(input_string)

    assert len(matches) == expected_matches_count, f"Expected {expected_matches_count} matches but got {len(matches)}"

    # New test input with new lines added
    input_string_with_new_lines = "Hello   \n\n   World!   This is   another test!   \n   "

    # Expected captures to clarify:
    # 1. "   " after "Hello"
    # 2. An empty line (will not necessarily count as a match)
    # 3. "   " between "World!" and "This"
    # 4. "   " between "is" and "another"
    # 5. "   " after "test!"
    expected_new_line_matches = 4  # 4 matches given clearer structure

    matches_with_new_lines = PRETTIFY_RE['DUPLICATES'].findall(input_string_with_new_lines)

    assert len(matches_with_new_lines) == expected_new_line_matches, f"Expected {expected_new_line_matches} but got {len(matches_with_new_lines)}"

# This test is designed to validate whether the code passes or fails based on regex behavior differences.