from string_utils.validation import is_camel_case

def test__is_camel_case():
    """
    Test the is_camel_case function to ensure it correctly identifies camel case strings.
    The test is designed to distinguish between the baseline and mutant. Specifically, it checks:
    - Empty strings should return False.
    - Valid camel case strings should return True.
    - Invalid camel case strings (including snake case and all lowercase/uppercase) should return False.

    The mutant introduces a bug by accepting non-camel case strings as valid, which should fail this test.
    """
    test_cases = [
        ("", False),  # Empty string should return False
        ("myVariable", True),  # Valid camel case should return True
        ("my_variable", False),  # Invalid camel case (snake case) should return False
        ("MyVariable", True),  # Valid camel case should return True
        ("myvariable", False),  # Invalid camel case (all lowercase) should return False
        ("MYVARIABLE", False),  # Invalid camel case (all uppercase) should return False
    ]
    
    for input_string, expected in test_cases:
        output = is_camel_case(input_string)
        print(f"input: '{input_string}' => output: {output}, expected: {expected}")
        assert output == expected