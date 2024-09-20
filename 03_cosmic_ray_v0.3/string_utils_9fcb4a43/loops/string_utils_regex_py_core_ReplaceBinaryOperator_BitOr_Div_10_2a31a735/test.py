import re
from string_utils._regex import PRETTIFY_RE

def test__PRETTIFY_RE():
    """The mutant trying to compile PRETTIFY_RE using '/' instead of '|' should raise an error."""

    # Access the regex pattern for 'SPACES_AROUND' in the PRETTIFY_RE dictionary
    correct_pattern = PRETTIFY_RE['SPACES_AROUND']
    
    # Ensure that the correct regex can be compiled successfully
    try:
        re.compile(correct_pattern.pattern, correct_pattern.flags)
    except Exception as e:
        assert False, f"Correct regex raised an exception: {e}"

    # The mutant regex should raise an error
    try:
        mutant_PRETTIFY_RE = re.compile(
            r'('
            r'(?<=")[^"]+(?=")|'  # quoted text ("hello world")
            r'(?<=\()[^)]+(?=\))'  # text in round brackets
            r')',
            re.MULTILINE / re.DOTALL  # This should raise an error
        )
        assert False, "Mutant regex compiled successfully (unexpected)"
    except Exception as e:
        assert "unsupported operand type(s) for &" in str(e), "Mutant did not fail appropriately"