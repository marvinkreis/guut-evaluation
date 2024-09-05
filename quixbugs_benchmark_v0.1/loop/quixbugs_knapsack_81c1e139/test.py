from knapsack import knapsack

def test__knapsack():
    output = knapsack(10, [(5, 10), (5, 10)])
    assert output == 20, "The correct knapsack implementation must return 20 for these inputs"