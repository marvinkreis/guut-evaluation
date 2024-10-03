from string_utils.manipulation import slugify

def test__slugify():
    """
    Test the slugify function to ensure it correctly formats a string by replacing spaces with a separator and does not allow 
    multiple consecutive separators. The input contains multiple words with spaces, which should be formatted correctly. 
    The mutant has introduced an error by incorrectly changing the '+' to '**', which will cause the regex to fail. 
    Consequently, when the input contains consecutive spaces (e.g., 'hello  world'), the output will differ between 
    the correct implementation and the mutant.
    """
    output = slugify('hello   world')
    assert output == 'hello-world'