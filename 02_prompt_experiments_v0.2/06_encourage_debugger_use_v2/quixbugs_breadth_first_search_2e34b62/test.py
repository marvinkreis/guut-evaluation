from breadth_first_search import breadth_first_search

def test__breadth_first_search():
    """The mutant version causes an infinite loop and raises an error due to the pop operation."""
    
    class Node:
        def __init__(self, value=None, successors=[]):
            self.value = value
            self.successors = successors

    # Setup nodes
    node_a = Node('A')
    node_b = Node('B')
    node_c = Node('C')
    
    # Define connections (node_a cannot reach node_c)
    node_a.successors = [node_b]
    node_b.successors = []  # B has no successors
    node_c.successors = []  # C has no successors

    startnode = node_a
    goalnode = node_c  # C is unreachable from A

    # Perform the correct breadth_first_search
    output = breadth_first_search(startnode, goalnode)
    
    # Verify that the output is as expected (False)
    assert output == False, "breadth_first_search should return False for unreachable nodes"