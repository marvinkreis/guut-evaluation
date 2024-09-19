from hanoi import hanoi

def test__hanoi():
    """Changing (start, end) to (start, helper) in hanoi would result in incorrect moves being recorded."""
    height = 2
    output = hanoi(height)
    # The expected correct output for height 2
    expected_output = [(1, 2), (1, 3), (2, 3)]
    
    assert output == expected_output, f"Expected steps to be {expected_output} but got {output}"