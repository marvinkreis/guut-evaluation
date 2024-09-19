from possible_change import possible_change

def test__possible_change():
    """Removing the 'not coins' check causes the mutant to raise an exception when calling with no coins."""
    output = possible_change([], 10)
    assert output == 0, "Expected output is 0 when there are no coins available."