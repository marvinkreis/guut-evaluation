from knapsack import knapsack

def test__knapsack():
    """Changing 'weight <= j' to 'weight < j' in knapsack would cause it to miss including items that exactly match the current weight capacity."""
    capacity = 100
    items = [(100, 50), (60, 10), (50, 8), (20, 4)]
    
    output = knapsack(capacity, items)
    assert output == 50, "knapsack must correctly include the item with weight equal to capacity"