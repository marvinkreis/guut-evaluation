from string_utils.manipulation import roman_encode

def test__roman_encode():
    # Testing the roman_encode function
    assert roman_encode(6) == 'VI'  # Correct output for 6
    assert roman_encode(7) == 'VII'  # Correct output for 7
    assert roman_encode(8) == 'VIII'  # Correct output for 8
    
    print("All tests passed!")

# Execute the test
test__roman_encode()