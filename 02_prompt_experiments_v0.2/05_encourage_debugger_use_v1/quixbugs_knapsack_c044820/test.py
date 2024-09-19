from knapsack import knapsack

def test__knapsack():
    """The mutant changes from 'weight <= j' to 'weight < j', causing it to miss optimal items."""
    capacity = 20
    items = [(15, 100), (10, 60), (5, 50)]
    
    output = knapsack(capacity, items)
    # The expected output should be 150 (using the item of weight 15 and value 100).
    assert output == 150, "The knapsack must calculate the correct maximum value."