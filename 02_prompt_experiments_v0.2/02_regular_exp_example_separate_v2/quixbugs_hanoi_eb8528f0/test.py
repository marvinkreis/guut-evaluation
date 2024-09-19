from hanoi import hanoi

def test__hanoi():
    """Changing the moves in hanoi from (start, end) to (start, helper) will result in incorrect behavior."""
    output = hanoi(1)
    assert output == [(1, 3)], "hanoi must move the disk from peg 1 to peg 3 with height of 1"