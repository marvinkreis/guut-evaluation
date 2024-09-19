from levenshtein import levenshtein

def test__levenshtein():
    output = levenshtein('electron', 'neutron')
    assert output == 3, "The Levenshtein distance should be 3 for 'electron' and 'neutron'"