from string_utils.manipulation import roman_encode

def test__roman_encode():
    # Test a number that should encode correctly
    assert roman_encode(100) == 'C'  # Should return 'C'
    assert roman_encode(200) == 'CC'  # Should return 'CC'
    assert roman_encode(300) == 'CCC'  # Should return 'CCC'
    assert roman_encode(400) == 'CD'  # Should return 'CD'
    assert roman_encode(500) == 'D'  # Should return 'D'
    assert roman_encode(600) == 'DC'  # Should return 'DC'
    assert roman_encode(700) == 'DCC'  # Should return 'DCC'
    assert roman_encode(800) == 'DCCC'  # Should return 'DCCC'
    assert roman_encode(900) == 'CM'  # Should return 'CM'
    assert roman_encode(1000) == 'M'  # Should return 'M'