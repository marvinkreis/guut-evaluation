from levenshtein import levenshtein

def test__levenshtein_different_start():
    """
    Test the Levenshtein distance when the starting characters of source and target differ.
    The input strings 'kitten' and 'sitting' will have a Levenshtein distance of 3 since
    they differ by three edits: changing 'k' to 's', 'e' to 'i', and adding 'g' to the end.
    """
    source = "kitten"
    target = "sitting"
    output = levenshtein(source, target)
    print(f"Levenshtein distance between '{source}' and '{target}' = {output}")
    assert output == 3  # Expected distance is 3