from knapsack import knapsack

def test__knapsack():
    """The mutant's change 'weight <= j' to 'weight < j' causes it to miss the maximum value from the item that perfectly fits the knapsack."""
    capacity = 100
    items = [(60, 10), (50, 8), (20, 4), (20, 4), (8, 3), (3, 2), (100, 50)]  # 100, 50 should give maximum value of 50
    output = knapsack(capacity, items)
    assert output == 50, "knapsack must return the maximum value considering all items"