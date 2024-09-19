from rpn_eval import rpn_eval

def test__rpn_eval():
    output = rpn_eval([3.0, 5.0, '+', 2.0, '/'])
    assert output == 4.0, "Expected 4.0 but got a different result"
    
    output_subtract = rpn_eval([5.0, 3.0, '-', 2.0, '/'])
    assert output_subtract == 1.0, "Expected 1.0 but got a different result"