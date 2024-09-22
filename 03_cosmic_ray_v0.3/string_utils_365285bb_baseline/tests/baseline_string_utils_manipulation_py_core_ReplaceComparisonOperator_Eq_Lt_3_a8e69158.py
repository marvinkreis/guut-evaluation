from string_utils.manipulation import compress

def test_compress_empty_string():
    try:
        compress('', encoding='utf-8')
        assert False, "Expected ValueError for empty input string, but no exception was raised."
    except ValueError as e:
        assert str(e) == 'Input string cannot be empty', f"Unexpected error message: {e}"

# Run the test
test_compress_empty_string()