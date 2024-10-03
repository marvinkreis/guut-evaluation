from possible_change import possible_change

def test__possible_change_empty_coins_kill_mutant():
    """
    Test the behavior of the possible_change function when the coins list is empty.
    The expectation is that there are no ways to make change for any positive total, 
    which should return 0 in the baseline, but cause a ValueError in the mutant.
    This test will effectively kill the mutant by revealing the robust handling 
    in the baseline with a positive total.
    """
    coins = []
    totals = [1, 10, 100]  # Testing with multiple positive totals

    for total in totals:
        output = possible_change(coins, total)
        assert output == 0  # Expecting 0 ways to make change