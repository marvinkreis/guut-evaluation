from string_utils.manipulation import prettify

def test__prettify():
    """This test checks that the mutant introduces unwanted spaces before 's in Saxon genitives."""
    input_string = "Sarah' s book is different from Paul's car, but Tim' s game is fun!"
    output = prettify(input_string)

    # The expected output
    expected_output = "Sarah's book is different from Paul's car, but Tim's game is fun!"

    # Overall equality check
    assert output == expected_output, "prettify output does not match expected formatting"

    # Assert that the actual character sequences do not include unintended spaces in the context of saxon genitives
    assert " s " not in output, "prettify should not incorrectly space before 's in Saxon genitives"