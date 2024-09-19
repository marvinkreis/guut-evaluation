from possible_change import possible_change

def test__possible_change_empty_coins():
    output = possible_change([], 10)
    assert output == 0, "Should be 0 ways to make change with no coins"