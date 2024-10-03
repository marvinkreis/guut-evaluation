from node import Node
from breadth_first_search import breadth_first_search

def test__breadth_first_search_reachability():
    """
    Test whether the function returns False when the goal node is unreachable from the start node. 
    This should pass for the Baseline implementation and fail for the Mutant implementation that gets stuck in an infinite loop.
    """
    # Create nodes
    node1 = Node(value="A")
    node2 = Node(value="B")  # This node will be unreachable

    # Setup the successors (node1 does not lead to node2)
    node1.successors = []  # node1 has no successors
    node2.successors = []   # and neither does node2

    output = breadth_first_search(node1, node2)
    print(f"output = {output}")
    assert output is False, "Expected the result to be False as the goal node is unreachable."