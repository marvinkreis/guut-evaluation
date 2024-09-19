from kheapsort import kheapsort

def test__kheapsort():
    input_array = [3, 2, 1, 5, 4]
    output = list(kheapsort(input_array, 2))
    assert output == sorted(input_array), "kheapsort must output the sorted array"