from bitcount import bitcount

def test__bitcount():
    output = bitcount(127)
    assert output == 7, f"Expected 7, but got {output} instead."