from levenshtein import levenshtein

def test__levenshtein_kill_mutant():
    """
    Test the Levenshtein distance between two strings with a common prefix. The input ('kitten', 'sitting') should yield a distance of 3 in the baseline, while the mutant should yield a distance of 7. This discrepancy occurs due to the mutant erroneously adding 1 when the first characters match, which should not happen.
    """
    source = "kitten"
    target = "sitting"
    output = levenshtein(source, target)
    assert output == 3