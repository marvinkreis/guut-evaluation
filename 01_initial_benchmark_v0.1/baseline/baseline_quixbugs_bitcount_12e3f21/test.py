from bitcount import bitcount

def test_bitcount():
    # Test with a known value
    assert bitcount(127) == 7  # 127 in binary is 1111111, which has 7 '1's
    assert bitcount(128) == 1   # 128 in binary is 10000000, which has 1 '1'
    
    # Test edge cases
    assert bitcount(0) == 0     # 0 in binary is 0, which has 0 '1's
    assert bitcount(1) == 1     # 1 in binary is 1, which has 1 '1'
    assert bitcount(15) == 4    # 15 in binary is 1111, which has 4 '1's