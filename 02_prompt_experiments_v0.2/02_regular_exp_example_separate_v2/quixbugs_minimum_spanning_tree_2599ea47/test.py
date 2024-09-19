from minimum_spanning_tree import minimum_spanning_tree

def test__minimum_spanning_tree():
    """The mutant's use of `update` instead of assignment causes a RuntimeError during iteration."""
    edges = {
        (1, 2): 10,
        (2, 3): 15,
        (3, 4): 10,
        (1, 4): 10
    }
    
    output = minimum_spanning_tree(edges)
    assert len(output) > 0, "The minimum spanning tree should contain edges"