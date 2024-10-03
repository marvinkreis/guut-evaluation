from string_utils.manipulation import reverse

def test__reverse():
    """
    This test checks whether the 'reverse' function correctly reverses the input string.
    The input 'hello' should return 'olleh'. The mutant changes the slicing logic 
    and returns the original string instead, which will fail this test.
    """
    output = reverse('hello')
    assert output == 'olleh'