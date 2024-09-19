from breadth_first_search import breadth_first_search
from node import Node

def test__breadth_first_search():
    """In the mutant, it will fail to handle paths correctly due to an infinite loop."""
    # Setup a simple graph where node A has no successors that lead to node B
    nodeA = Node(value='A', successors=[])
    nodeB = Node(value='B', successors=[])

    # Correct call should return False since there is no path from A to B
    output = breadth_first_search(nodeA, nodeB)
    assert output is False, "Expecting to return False when there is no path from A to B"