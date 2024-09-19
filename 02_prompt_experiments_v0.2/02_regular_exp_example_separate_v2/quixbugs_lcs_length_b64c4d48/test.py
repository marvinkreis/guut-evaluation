from lcs_length import lcs_length

def test__lcs_length():
    """The mutant's change in how lcs_length calculates common substrings will lead to incorrect outputs."""
    
    # Test case 1: witch and sandwich
    output1 = lcs_length('witch', 'sandwich')
    assert output1 == 2, f"Expected 2 but got {output1} for input ('witch', 'sandwich')"
    
    # Test case 2: meow and homeowner
    output2 = lcs_length('meow', 'homeowner')
    assert output2 == 4, f"Expected 4 but got {output2} for input ('meow', 'homeowner')"