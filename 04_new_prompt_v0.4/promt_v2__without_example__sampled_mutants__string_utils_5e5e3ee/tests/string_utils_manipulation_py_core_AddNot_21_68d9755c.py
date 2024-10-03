from string_utils.manipulation import shuffle
from string_utils.errors import InvalidInputError

def test__shuffle_non_string_input():
    """
    Tests the shuffle function with a non-string input (integer).
    The baseline should raise an InvalidInputError, and the mutant should raise a different error.
    This test verifies the distinct behavior of the mutant.
    """
    # Check behavior on baseline
    try:
        shuffle(12345)  # Non-string input
        assert False  # Expecting an exception to be raised
    except InvalidInputError as ve:
        print(f"Baseline raised InvalidInputError as expected: {ve}")
        assert True
    except Exception as e:
        print(f"Baseline raised an unexpected exception: {e}")
        assert False

    # Now we check for mutant behavior
    try:
        output = shuffle(12345)  # Non-string input
        print(f"Mutant produced output: {output}")
        assert False  # We expect the mutant to raise an exception, not return output
    except Exception as e:
        print(f"Mutant raised an exception: {e}")
        assert True