from shunting_yard import shunting_yard

def test__shunting_yard():
    output = shunting_yard([10, '-', 5, '-', 2])
    assert output == [10, 5, '-', 2, '-'], "The output should match the expected RPN."