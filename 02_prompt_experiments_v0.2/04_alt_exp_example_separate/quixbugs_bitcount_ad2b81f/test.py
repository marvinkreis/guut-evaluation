from bitcount import bitcount

def test__bitcount():
    """The mutant changes the bitwise AND to XOR, leading to incorrect behavior."""
    output = bitcount(3)
    assert output == 2, "bitcount must correctly count the number of 1-bits"