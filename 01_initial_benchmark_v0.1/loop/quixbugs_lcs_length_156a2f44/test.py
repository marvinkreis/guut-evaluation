from lcs_length import lcs_length

def test__lcs_length():
    assert lcs_length('witch', 'sandwich') == 2, "Expected length of longest common substring to be 2."
    assert lcs_length('meow', 'homeowner') == 4, "Expected length of longest common substring to be 4."