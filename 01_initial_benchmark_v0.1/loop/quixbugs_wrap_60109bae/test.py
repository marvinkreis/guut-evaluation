from wrap import wrap

def test__wrap():
    input_text = "This is a long string that needs to be wrapped correctly."
    input_cols = 20
    output = wrap(input_text, input_cols)
    
    assert len(output) > 3, "Output should contain more than 3 lines."
    assert output[-1] == " correctly.", "The last line should correctly append the final segment."