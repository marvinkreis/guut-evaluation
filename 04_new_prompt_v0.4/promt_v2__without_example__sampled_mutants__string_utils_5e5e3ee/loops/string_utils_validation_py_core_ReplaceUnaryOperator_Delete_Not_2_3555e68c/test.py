from string_utils.validation import is_url

def test__is_url_mutant_killer():
    """
    Test the is_url function to confirm that it correctly identifies valid URLs.
    This test will expose the mutant, as it should return True for valid HTTP and HTTPS URLs.
    """
    assert is_url('http://example.com') == True, "Expected True for HTTP URL"
    assert is_url('https://example.com') == True, "Expected True for HTTPS URL"