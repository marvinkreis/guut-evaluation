import re
from string_utils._regex import WORDS_COUNT_RE

def test__words_count():
    """Testing WORDS_COUNT_RE for nuanced edge cases to discover mutant behavior."""
    
    # Group of normal cases and expected outputs
    test_strings = [
        "Cats and dogs.",                   # Expected: ['Cats', 'and', 'dogs']
        "!@#$%^&*()",                       # Expected: []
        "123 test test! 456",               # Expected: ['test', 'test']
        "word1.......word2",                # expected: ['word1.......word2']
        "   ",                              # Expected: []
        "It's a beautiful day.",            # Expected: ['It', 's', 'a', 'beautiful', 'day']
        "\n \n word1\nword2\n",            # Expected: ['word1', 'word2']
    ]
    
    for input_string in test_strings:
        correct_output = WORDS_COUNT_RE.findall(input_string)
        print(f"Correct output for '{input_string}': {correct_output}")

        mutant_output = WORDS_COUNT_RE.findall(input_string)
        print(f"Mutant output for '{input_string}': {mutant_output}")

# Execute the test confirming mutant behavior
test__words_count()