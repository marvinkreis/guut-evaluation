from string_utils.validation import is_isbn_10

def test__is_isbn_10_kill_mutant():
    """
    Test a known valid ISBN-10 number that is expected to return True in the baseline
    but False in the mutant due to the logical modification in the is_isbn_10 method.
    The test checks the ISBN-10 number '1112223339'.
    """
    valid_isbn_10_alt = '1112223339'  # Known valid ISBN-10
    output = is_isbn_10(valid_isbn_10_alt)
    print(f"output = {output}")
    assert output == True  # Expecting True in the baseline, False in the mutant