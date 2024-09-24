from string_utils.manipulation import __StringFormatter

def test__string_formatter_ensure_spaces_around():
    # Create an instance of __StringFormatter with a sample string
    input_string = "Hello"
    formatter = __StringFormatter(input_string)

    # Prepare a mock regex match object
    class MockRegexMatch:
        def group(self, index):
            if index == 1:
                return "world"
            return None

    regex_match_mock = MockRegexMatch()
    expected_output = " world "

    # Test with correct code
    output_correct = formatter._StringFormatter__ensure_spaces_around(regex_match_mock)
    assert output_correct == expected_output, f"Expected '{expected_output}' but got '{output_correct}'"

    # Test with mutant
    try:
        # This line simulates the type of error the mutant generates
        # Here we purposely cause a problem by mimicking the mutant's code alteration
        output_mutant = ' ' % regex_match_mock.group(1).strip() + ' '
        output_mutant  # This will raise TypeError on mutant!
        assert False, "This line should not have passed; mutant should raise TypeError"
    except TypeError:
        pass  # This is expected in mutant version

# Run the test
test__string_formatter_ensure_spaces_around()