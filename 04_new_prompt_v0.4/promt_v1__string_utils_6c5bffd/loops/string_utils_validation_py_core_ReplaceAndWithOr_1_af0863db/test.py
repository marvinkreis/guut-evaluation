from string_utils.validation import is_integer

def test__is_integer_mutant_killer():
    """
    Test that verifies the correct identification of integers. The mutant version incorrectly validates decimal inputs as integers.
    Test input '42.0' should return False for Baseline but True for the Mutant.
    """
    assert is_integer('42.0') == False