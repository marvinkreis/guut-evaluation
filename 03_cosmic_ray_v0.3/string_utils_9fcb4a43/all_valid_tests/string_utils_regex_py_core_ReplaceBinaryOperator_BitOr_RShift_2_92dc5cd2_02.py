import re
from string_utils._regex import WORDS_COUNT_RE

def test__words_count():
    """Testing WORDS_COUNT_RE against edge cases and malformed regex to differentiate mutant behavior."""
    
    # Basic valid test cases expecting match results
    test_strings = [
        "Hello there! How are you?",  # Normal case
        "!@#$%^&*()",                  # Special characters; expect no matches
        "Today is a sunny day!",       # Standard sentence
    ]
    
    for test_string in test_strings:
        output = WORDS_COUNT_RE.findall(test_string)
        assert output is not None, "Output should not be None."
        assert isinstance(output, list), "Output should be a list."
    
    # Inducing an error in regex compilation
    faulty_pattern = 'invalid regex ['  # Intentionally malformed regex
    
    # Test for correct implementation's handling of faulty regex
    try:
        re.compile(faulty_pattern) 
        assert False, "Expected a compilation error for the faulty regex!"
    except re.error as e:
        print(f"Correct implementation caught an error: {str(e)}")

    # Now check the mutant in the same faulty regex context
    try:
        mutant_output = re.compile(faulty_pattern) 
        assert False, "Expected mutant implementation to fail due to faulty regex!"
    except Exception as e:  # Catching any exceptions, tougher on mutants
        print(f"Mutant raised an exception with faulty regex: {str(e)}")
        assert isinstance(e, re.error), "Mutant should raise a re.error for regex compilation."

# Run the test
test__words_count()