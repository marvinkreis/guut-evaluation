from bucketsort import bucketsort

def test__bucketsort():
    input_data = [3, 1, 4, 1, 5]
    expected_output = [1, 1, 3, 4, 5]
    output = bucketsort(input_data, 6)
    assert output == expected_output, f"Expected {expected_output}, got {output}"