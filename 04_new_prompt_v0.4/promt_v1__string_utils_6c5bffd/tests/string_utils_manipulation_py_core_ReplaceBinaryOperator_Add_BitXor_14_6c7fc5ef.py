from string_utils.manipulation import camel_case_to_snake

def test__camel_case_to_snake():
    """
    Test the camel_case_to_snake function to ensure it converts camel case to snake case correctly.
    The expected outputs confirm that the function produces valid snake case strings for standard inputs. 
    The test also ensures the mutant is detected, as it will raise a TypeError due to an incorrect XOR operation.
    """
    test_cases = [
        ('CamelCaseString', 'camel_case_string'),
        ('AnotherExampleHere', 'another_example_here'),
        ('YetAnotherOne', 'yet_another_one'),
        ('SimpleTest', 'simple_test')
    ]
    
    for input_string, expected_output in test_cases:
        output = camel_case_to_snake(input_string)
        assert output == expected_output