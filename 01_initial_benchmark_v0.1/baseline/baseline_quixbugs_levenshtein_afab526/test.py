from levenshtein import levenshtein

def test__levenshtein():
    source = "kitten"
    target = "sitten"
    # The correct Levenshtein distance is 1 (change 'k' to 's')
    expected_distance = 1
    assert levenshtein(source, target) == expected_distance, "Test failed: Expected distance doesn't match the computed"

    source = "flaw"
    target = "lawn"
    # The correct Levenshtein distance is 2 (change 'f' to 'l' and add 'n')
    expected_distance = 2
    assert levenshtein(source, target) == expected_distance, "Test failed: Expected distance doesn't match the computed"