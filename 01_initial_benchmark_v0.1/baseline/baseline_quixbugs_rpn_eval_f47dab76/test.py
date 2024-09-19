from rpn_eval import rpn_eval

def test_rpn_eval_subtraction_order():
    # This test will check if the function correctly evaluates
    # 5.0 - 3.0, which should give us 2.0 in proper order.
    result = rpn_eval([5.0, 3.0, '-'])
    assert result == 2.0, f"Expected 2.0 but got {result}"

    # Now, let’s validate the behavior of the mutant.
    # The mutant will evaluate it as 3.0 - 5.0, which should give us -2.0.
    # Since we cannot execute the mutant directly here, knowing its behavior is enough.
    # The assertion above will pass only if the original code works correctly.