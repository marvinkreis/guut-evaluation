from flatten import flatten

def test__flatten():
    """
    Test the flatten function to ensure it correctly flattens a nested list
    into a single list of non-list objects. The input represents a nested list
    structure. The mutant will fail since it outputs generator objects instead 
    of a flat list of values.
    """
    input_data = [[1, [], [2, 3]], [[4]], 5]
    expected_output = [1, 2, 3, 4, 5]
    output = list(flatten(input_data))
    assert output == expected_output, f"Expected {expected_output} but got {output}"