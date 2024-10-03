from possible_change import possible_change

def test__possible_change_kills_mutant():
    """
    Test the possible_change function with a list of coins and a positive total to ensure that both baseline and mutant versions produce different results. 
    The baseline should calculate distinct combinations correctly, while the mutant will raise a ValueError due to its lack of a condition checking for empty coins.
    """
    output = possible_change([1, 2, 5], 5)
    assert output == 4  # Expected combinations: (5), (2,2,1), (2,1,1,1), (1,1,1,1,1)