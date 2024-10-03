from string_utils.manipulation import strip_margin
from string_utils.errors import InvalidInputError  # Importing the error explicitly

def test__strip_margin_numeric_string():
    """
    Test the strip_margin function with a numeric string input. The baseline should process it normally,
    while the mutant is expected to raise an InvalidInputError due to its incorrect input validation logic.
    """
    # Testing with numeric string input
    output_baseline = strip_margin('12345')  # Should pass without error in the baseline
    
    # We expect this to raise an InvalidInputError in the mutant
    try:
        strip_margin('12345')
        output_mutant = "No error raised"
    except InvalidInputError as e:
        output_mutant = "InvalidInputError"
    except Exception as e:
        output_mutant = type(e).__name__

    print(f"Baseline output: '{output_baseline}'")
    print(f"Mutant output: {output_mutant}")