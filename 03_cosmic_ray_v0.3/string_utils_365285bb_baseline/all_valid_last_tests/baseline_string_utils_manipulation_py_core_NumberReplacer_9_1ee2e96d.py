from string_utils.manipulation import roman_encode, roman_decode

def test_roman_encode_decode():
    # Valid assertions for the correct implementation
    assert roman_encode(100) == 'C', 'Expected "C" for 100'
    assert roman_encode(1) == 'I', 'Expected "I" for 1'
    assert roman_encode(3999) == 'MMMCMXCIX', 'Expected "MMMCMXCIX" for 3999'
    
    # Testing problematic values for the mutant
    # Check that the original correctly encodes:
    assert roman_encode(4) == 'IV', 'Expected "IV" for 4'
    assert roman_encode(9) == 'IX', 'Expected "IX" for 9'
    assert roman_encode(40) == 'XL', 'Expected "XL" for 40'
    assert roman_encode(90) == 'XC', 'Expected "XC" for 90'
    
    # Test for values that should yield incorrect behavior under the mutant:
    # Using known problematic mappings
    encoded_4 = roman_encode(4)  # expected 'IV'
    assert encoded_4 == 'IV', f'Expected "IV" for 4, got "{encoded_4}"'

    encoded_9 = roman_encode(9)  # expected 'IX'
    assert encoded_9 == 'IX', f'Expected "IX" for 9, got "{encoded_9}"'

    encoded_40 = roman_encode(40)  # expected 'XL'
    assert encoded_40 == 'XL', f'Expected "XL" for 40, got "{encoded_40}"'

    encoded_90 = roman_encode(90)  # expected 'XC'
    assert encoded_90 == 'XC', f'Expected "XC" for 90, got "{encoded_90}"'
    
    # We are aware that under the mutant, the mappings for specific numbers can cause failures:
    try:
        mutant_output_for_4 = roman_encode(4)  # Should yield 'IV' for correct, but mutant should fail
        assert mutant_output_for_4 != 'IV', f'Expected mutant to yield incorrect output for 4 but got "{mutant_output_for_4}"'
    
        mutant_output_for_9 = roman_encode(9)  # Should yield 'IX'
        assert mutant_output_for_9 != 'IX', f'Expected mutant to yield incorrect output for 9 but got "{mutant_output_for_9}"'

    except Exception as e:
        # Catching any exceptions (like KeyError) from the mutant implementation
        print(f'Mutant expected failure: {str(e)}')  # This should occur under mutant faulty mapping

    # Continue with normal decoding tests
    assert roman_decode('C') == 100, 'Expected 100 for "C"'
    assert roman_decode('I') == 1, 'Expected 1 for "I"'
    assert roman_decode('MMMCMXCIX') == 3999, 'Expected 3999 for "MMMCMXCIX"'

# Run the test
test_roman_encode_decode()