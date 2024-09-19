from find_first_in_sorted import find_first_in_sorted

def test__find_first_in_sorted():
    """Changing the while loop condition to 'lo <= hi' will cause an IndexError on certain inputs."""
    arr = [1, 2, 3, 4, 5]
    x = 6
    output = find_first_in_sorted(arr, x)
    assert output == -1, "The function should return -1 if the value is not found."