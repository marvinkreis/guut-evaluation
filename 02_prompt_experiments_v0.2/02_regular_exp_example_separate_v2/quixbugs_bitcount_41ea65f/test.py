from bitcount import bitcount

def test__bitcount():
    """The mutant's use of XOR instead of AND causes it to count infinitely for certain inputs."""
    assert bitcount(127) == 7, "bitcount(127) should return 7"
    assert bitcount(128) == 1, "bitcount(128) should return 1"