from string_utils.manipulation import prettify

def test__prettify():
    # Testing the prettify function to ensure it correctly formats the string.
    input_string = ' hello  world '
    expected_output = 'Hello world'
    
    # The test case should pass under the original implementation
    assert prettify(input_string) == expected_output
    
    # To detect the mutant, I will change the input string 
    # that should produce a different output if the mutant is present.
    mutant_input = ' space   test  '
    mutant_expected_output = 'Space test'  # This is what we'd expect from correct code
    assert prettify(mutant_input) == mutant_expected_output
    
    # If the mutant is present, the assertion should fail