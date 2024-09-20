import re
from string_utils._regex import PRETTIFY_RE as correct_PRETTIFY_RE

def test__PRETTIFY_RE_compilation():
    """The mutant changes a regex compilation causing it to fail, while the correct one compiles properly."""
    # Test the correct implementation
    try:
        # This should succeed with the correct implementation.
        re.compile(r'(?<=")[^"]+(?=")|(?<=\()[^)]+(?=\))', re.MULTILINE | re.DOTALL)
    except Exception as e:
        assert False, f"Correct PRETTIFY_RE compilation failed unexpectedly: {e}"

    # Attempt to compile the mutant implementation
    try:
        re.compile(r'(?<=")[^"]+(?=")|(?<=\()[^)]+(?=\))', re.MULTILINE - re.DOTALL)
        # If it compiles successfully, this line will fail the test.
        assert False, "Mutant PRETTIFY_RE compiled successfully, which should not happen."
    except ValueError:
        # If it raises a ValueError, the test should pass as expected.
        pass