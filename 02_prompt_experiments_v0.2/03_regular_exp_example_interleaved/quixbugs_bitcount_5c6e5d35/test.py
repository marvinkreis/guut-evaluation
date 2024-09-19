from bitcount import bitcount

def test__bitcount():
    """The mutant changes '&' to '^', leading to incorrect or infinite results."""
    assert bitcount(3) == 2, "Expected 2 for input 3"
    assert bitcount(7) == 3, "Expected 3 for input 7"