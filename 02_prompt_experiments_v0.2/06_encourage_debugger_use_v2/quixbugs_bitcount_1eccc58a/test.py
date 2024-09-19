from bitcount import bitcount

def test__bitcount():
    """Changing '&=' to '^=' in bitcount will cause infinite loops or incorrect counts for any n > 0."""
    assert bitcount(0) == 0, "bitcount(0) must return 0"
    assert bitcount(1) == 1, "bitcount(1) must return 1"
    assert bitcount(2) == 1, "bitcount(2) must return 1"
    assert bitcount(3) == 2, "bitcount(3) must return 2"
    assert bitcount(4) == 1, "bitcount(4) must return 1"
    assert bitcount(255) == 8, "bitcount(255) must return 8"