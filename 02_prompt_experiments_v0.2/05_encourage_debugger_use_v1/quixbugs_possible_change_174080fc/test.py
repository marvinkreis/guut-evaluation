from possible_change import possible_change

def test__possible_change():
    """Removing the condition `not coins` in possible_change will lead to ValueErrors for some inputs."""
    
    # Test case where coins is empty and total > 0
    assert possible_change([], 5) == 0, "Should return 0 as there are no ways to make change."
    
    # Assert a recognized valid input that yields a computable output
    assert possible_change([1, 2], 3) == 2, "Should return 2 ways to make change for 3."
    
    # Test case where coins is empty and total is 0
    assert possible_change([], 0) == 1, "Should return 1 as there is one way to make no change."