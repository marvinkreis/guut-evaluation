from node import Node
from breadth_first_search import breadth_first_search

def test__breadth_first_search():
    """The mutant version fails on unreachable nodes; it should return False for a non-existent path."""
    # Create a single node that has no successors
    start_node = Node(value='A', successors=[])
    goal_node = Node(value='B', successors=[])

    output = breadth_first_search(start_node, goal_node)
    assert output == False, "breadth_first_search must return False when the goal node is unreachable"