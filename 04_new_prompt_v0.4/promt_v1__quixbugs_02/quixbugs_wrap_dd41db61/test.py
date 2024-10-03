from wrap import wrap

def test__wrap_mutant_killing():
    """
    This test checks the behavior of the wrap function with a specific text input, 
    aimed to reveal the mutant's failure to include a correctly wrapped last line. 
    The baseline includes the final segment of text that fits within the column width, 
    while the mutant omits it, leading to an assertion failure. 
    """
    text = "The sun sets over the horizon."
    cols = 30
    output = wrap(text, cols)
    expected_output = ["The sun sets over the horizon."]
    assert output == expected_output