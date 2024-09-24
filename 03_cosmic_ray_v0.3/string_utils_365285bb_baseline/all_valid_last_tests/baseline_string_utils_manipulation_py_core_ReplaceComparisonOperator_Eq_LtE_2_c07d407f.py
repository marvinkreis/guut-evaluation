from string_utils.manipulation import roman_encode

def test_roman_encode():
    # Assert basic outputs to verify correctness
    assert roman_encode(1) == 'I'      # 'I'
    assert roman_encode(2) == 'II'     # 'II'
    assert roman_encode(3) == 'III'    # 'III'
    assert roman_encode(4) == 'IV'     # 'IV'
    assert roman_encode(5) == 'V'      # 'V' - This is a critical point for the mutant!

    # Essential for identifying the mutant:
    assert roman_encode(6) == 'VI'     # 'VI' - Expected output, check for failure here

    # Further checks for robustness
    assert roman_encode(7) == 'VII'    # 'VII'
    assert roman_encode(8) == 'VIII'   # 'VIII'
    assert roman_encode(9) == 'IX'     # 'IX'
    assert roman_encode(10) == 'X'     # 'X'
    assert roman_encode(11) == 'XI'    # 'XI'
    assert roman_encode(12) == 'XII'   # 'XII'
    assert roman_encode(13) == 'XIII'  # 'XIII'
    assert roman_encode(14) == 'XIV'   # 'XIV'
    assert roman_encode(15) == 'XV'    # 'XV'
    assert roman_encode(16) == 'XVI'   # 'XVI'
    assert roman_encode(17) == 'XVII'  # 'XVII'
    assert roman_encode(18) == 'XVIII' # 'XVIII'
    assert roman_encode(19) == 'XIX'   # 'XIX'
    assert roman_encode(20) == 'XX'    # 'XX'
    
    # Check that upper limits remain correct
    assert roman_encode(3999) == 'MMMCMXCIX'  # Ensure the entire range holds
