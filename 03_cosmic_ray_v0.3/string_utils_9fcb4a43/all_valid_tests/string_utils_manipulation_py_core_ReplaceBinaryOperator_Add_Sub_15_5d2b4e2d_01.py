from string_utils.manipulation import slugify

def test__slugify():
    """The mutant introduces an error in slugify due to an invalid regex operation."""
    output = slugify("Test input string!")
    assert output == "test-input-string", "slugify must return a valid slug"