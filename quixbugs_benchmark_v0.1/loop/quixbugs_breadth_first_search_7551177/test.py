from breadth_first_search import breadth_first_search
from node import Node

def test__breadth_first_search():
    # Setting up test nodes
    node_a = Node(value='A')
    node_b = Node(value='B')

    # No connection between A and B
    node_a.successors = []

    # Testing unreachable goal node
    result = breadth_first_search(node_a, node_b)
    assert result == False, "breadth_first_search must return False when no path exists."