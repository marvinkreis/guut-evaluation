from minimum_spanning_tree import minimum_spanning_tree

def test__minimum_spanning_tree():
    weight_by_edge = {
        (1, 2): 10,
        (2, 3): 15,
        (3, 4): 10,
        (1, 4): 10
    }
    output = minimum_spanning_tree(weight_by_edge)
    assert output == {(1, 2), (3, 4), (1, 4)}, "Minimum spanning tree does not match expected output."