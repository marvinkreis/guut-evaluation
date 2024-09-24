from string_utils.manipulation import roman_encode

def test_roman_encode_detection():
    # Test the encoding of the number 20 (which should yield 'XX')
    expected_output = 'XX'
    actual_output = roman_encode(20)
    assert actual_output == expected_output, f"Expected {expected_output} but got {actual_output}"

    # Test the encoding of the number 30 (which should yield 'XXX')
    expected_output = 'XXX'
    actual_output = roman_encode(30)
    assert actual_output == expected_output, f"Expected {expected_output} but got {actual_output}"

    # Test the encoding of the number 40 (which should yield 'XL')
    expected_output = 'XL'
    actual_output = roman_encode(40)
    assert actual_output == expected_output, f"Expected {expected_output} but got {actual_output}"

    # Test the encoding of the number 4 (which should yield 'IV')
    expected_output = 'IV'
    actual_output = roman_encode(4)
    assert actual_output == expected_output, f"Expected {expected_output} but got {actual_output}"

    # Test the encoding of the number 1 (which should yield 'I')
    expected_output = 'I'
    actual_output = roman_encode(1)
    assert actual_output == expected_output, f"Expected {expected_output} but got {actual_output}"

# Run the test
if __name__ == "__main__":
    test_roman_encode_detection()
    print("All tests passed!")