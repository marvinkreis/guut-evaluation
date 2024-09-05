from max_sublist_sum import max_sublist_sum

def test__max_sublist_sum():
    output = max_sublist_sum([4, -5, 2, 1, -1, 3])
    assert output == 5, "max_sublist_sum must return the correct maximum sublist sum"