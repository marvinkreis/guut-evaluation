from string_utils.manipulation import roman_encode

def test_roman_encode_mutant_killing():
    """
    Test the roman_encode function with valid single-digit inputs (1, 2, 3).
    The baseline will return 'I', 'II', and 'III',
    while the mutant will raise a TypeError because of the left shift operation on a string.
    """
    results = []
    for num in [1, 2, 3]:
        try:
            output = roman_encode(num)
            results.append((num, output))
        except Exception as e:
            results.append((num, str(e)))

    # Validate the results for baseline behavior
    assert results[0] == (1, 'I'), f"Expected 'I', got {results[0][1]}"
    assert results[1] == (2, 'II'), f"Expected 'II', got {results[1][1]}"
    assert results[2] == (3, 'III'), f"Expected 'III', got {results[2][1]}"