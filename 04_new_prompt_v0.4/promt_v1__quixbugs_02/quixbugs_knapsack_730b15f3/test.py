from knapsack import knapsack

def test__knapsack():
    """
    Test case to ensure that the knapsack function with items [(30, 20), (20, 18), (50, 25)] returns 38 for capacity 50.
    The mutant implementation fails to account for the maximum value combinations when an item’s weight equals the remaining capacity.
    """
    capacity = 50
    items = [(30, 20), (20, 18), (50, 25)]
    output = knapsack(capacity, items)
    assert output == 38, f"Expected output 38, but got {output}"