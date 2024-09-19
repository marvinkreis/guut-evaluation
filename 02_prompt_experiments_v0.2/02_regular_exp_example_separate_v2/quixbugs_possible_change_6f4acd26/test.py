from possible_change import possible_change

def test__possible_change_empty_coins():
    """The mutant fails on empty coins list by raising a ValueError,
       while the correct implementation returns 0."""
    
    # This should return 0 for correct implementation
    output = possible_change([], 5)
    assert output == 0, f"Expected 0, but got {output}"

# Execute the test
test__possible_change_empty_coins()