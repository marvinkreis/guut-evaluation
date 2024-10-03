from string_utils.generation import roman_range

def test__roman_range_invalid_configuration_mutant_kill():
    """
    Test the roman_range function with a configuration that should produce
    an incorrect iteration when using a backward step that can potentially
    hit the stopping point incorrectly managed by the mutant.
    Using parameters: start = 4, stop = 1, step = -3. 
    The expected output should directly yield 'IV' before reaching the invalid state.
    The mutant is expected to cause a failure when allowed to iterate further.
    """
    output = list(roman_range(stop=1, start=4, step=-3))
    print("Output:", output)  # Expected: ['IV', 'I']
    assert output == ['IV', 'I'], "Output does not match the expected sequence."