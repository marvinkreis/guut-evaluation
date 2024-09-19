from knapsack import knapsack

def test__knapsack():
    """Changing '<=' to '<' in knapsack prevents the knapsack from using items that exactly match the capacity."""
    capacity = 100
    items = [(100, 100), (60, 10), (50, 8), (20, 4), (20, 4), (8, 3), (3, 2)]
    output = knapsack(capacity, items)
    assert output == 100, f"expected 100 but got {output}"