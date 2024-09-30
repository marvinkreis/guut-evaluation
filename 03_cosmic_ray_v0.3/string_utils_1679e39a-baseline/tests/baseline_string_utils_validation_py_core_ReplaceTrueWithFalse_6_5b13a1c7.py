from string_utils.validation import is_isbn

def test__is_isbn():
    """
    This test checks whether the 'normalize' parameter affects the behavior of the function by passing an ISBN-10 with hyphens. 
    The mutant changes the default value of 'normalize' to False, which will cause this test to fail as it expects correct
    normalization and validation of the ISBN when hyphens are present.
    """
    output = is_isbn('150-6715214')  # Should return True when normalize=True (default)
    assert output is True