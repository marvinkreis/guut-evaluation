from possible_change import possible_change

def test__possible_change():
    """The mutant removing the check for an empty coin list will likely yield different results."""
    # Testing scenarios with no coins and a positive total
    assert possible_change([], 10) == 0, "With no coins and a positive total, the return should be 0."

    # Testing scenarios with coins and a zero total
    assert possible_change([1], 0) == 1, "With coin 1 and total 0, the return should be 1."
    
    # Testing again with no coins but checking a zero total
    assert possible_change([], 0) == 1, "With no coins and total 0, there should be 1 way (using 0 coins)."