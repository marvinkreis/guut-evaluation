from string_utils.validation import is_slug

def test__is_slug_mutant_killing():
    """
    Test is_slug function with various inputs to confirm the mutant behavior.
    The input '' and ' ' should return False on Baseline (valid check),
    but True on Mutant. This will confirm the differences between Baseline and Mutant.
    """
    assert is_slug('') == False  # Baseline should return False
    assert is_slug(' ') == False  # Baseline should return False