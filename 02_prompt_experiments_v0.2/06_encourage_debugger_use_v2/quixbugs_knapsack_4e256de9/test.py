from knapsack import knapsack

def test__knapsack():
    """Changing '<=' to '<' in knapsack would lead to missing optimal items that fit the capacity."""
    output = knapsack(10, [(5, 10), (5, 15), (10, 5)])
    assert output == 25, "knapsack must return the maximum value of 25"
    
    output = knapsack(15, [(6, 10), (4, 5), (5, 8), (9, 12), (10, 15)])
    assert output == 23, "knapsack must return the maximum value of 23"