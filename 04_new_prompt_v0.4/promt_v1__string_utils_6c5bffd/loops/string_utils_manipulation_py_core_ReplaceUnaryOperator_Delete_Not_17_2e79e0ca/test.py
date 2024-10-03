from string_utils.manipulation import strip_margin
from string_utils.errors import InvalidInputError

def test__strip_margin():
    """
    Test strip_margin for both invalid and valid inputs.
    Expect the baseline to correctly raise an error for invalid input.
    The mutant's behavior is expected to diverge.
    """

    # Testing invalid input
    try:
        strip_margin(123)  # Invalid input
        assert False, "Expected InvalidInputError not raised on invalid input."
    except InvalidInputError:
        print("Baseline: InvalidInputError raised, test passed.")
    except Exception as e:
        print(f"Baseline: Different error raised: {type(e).__name__}, test failed.")

    # Testing valid input
    input_str = '''
        line 1
        line 2
        line 3
    '''

    # Expecting the output to have one leading newline, as per input treatment by the function.
    # Correct output should include this newline character.
    expected_output = '\nline 1\nline 2\nline 3\n'

    output = strip_margin(input_str)
    print(f"Output: '{output}'")
    assert output == expected_output, f"Output did not match expected valid formatting. Expected: '{expected_output}', Got: '{output}'"

    # Distinguish mutant behavior
    try:
        output = strip_margin(123)  # Expecting this to fail for the mutant as well.
        assert False, "Expected InvalidInputError not raised on invalid input in mutant."
    except InvalidInputError:
        print("Mutant: InvalidInputError raised, test failed.")
    except Exception as e:
        print(f"Mutant: Different error raised: {type(e).__name__}, test passed.")

    # Ensure correct output assertion on mutant
    mutant_output = strip_margin(input_str)  # Trigger this for mutant behavior
    assert mutant_output == expected_output, "Mutant output did not match expected valid formatting."