from knapsack import knapsack

def test__knapsack():
    """
    Test whether the function correctly includes an item when its weight exactly matches the remaining capacity.
    The input represents a knapsack with capacity 100 and an item with weight 100 and value 100.
    This test checks whether the item is included when its weight equals the knapsack's current capacity.
    The mutant should return a different value since it uses '<' instead of '<='.
    """
    output = knapsack(100, [(100, 100), (50, 60), (10, 20)])
    assert output == 100  # Baseline should include the item with weight 100