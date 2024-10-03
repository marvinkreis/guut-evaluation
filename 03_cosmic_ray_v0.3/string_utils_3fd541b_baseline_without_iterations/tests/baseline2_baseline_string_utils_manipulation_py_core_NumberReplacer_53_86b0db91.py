from string_utils.manipulation import compress, InvalidInputError

def test_compress_empty_string():
    try:
        compress('')
    except ValueError as e:
        assert str(e) == 'Input string cannot be empty'
    else:
        assert False, "Expected ValueError not raised"

# Run the test
test_compress_empty_string()