from string_utils.validation import is_isbn

def test_isbn_initialization():
    # 1. Test valid ISBN 13 with normalization -> should return True
    assert is_isbn('978-3-16-148410-0', normalize=True), "Expected True for valid ISBN 13 with normalization"

    # 2. Test valid ISBN 13 without hyphens and normalization -> should return True
    assert is_isbn('9783161484100', normalize=False), "Expected True for valid ISBN 13 without hyphens, normalization off"

    # 3. Test valid ISBN 13 with hyphens but no normalization -> should return False in correct implementation but True in mutant
    assert is_isbn('978-3-16-148410-0', normalize=False) == False, "Expected False for valid ISBN 13 with hyphens, normalization off"

    # 4. Test valid ISBN 10 with normalization -> should return True
    assert is_isbn('1506715214', normalize=True), "Expected True for valid ISBN 10 with normalization"

    # 5. Test valid ISBN 10 without hyphens and normalization -> should return True
    assert is_isbn('1506715214', normalize=False), "Expected True for valid ISBN 10 without normalization"

    # 6. Test valid ISBN 10 with hyphens and normalization turned off -> should return False in correct implementation but True in mutant
    assert is_isbn('150-6715214', normalize=False) == False, "Expected False for valid ISBN 10 with hyphens and normalization off"