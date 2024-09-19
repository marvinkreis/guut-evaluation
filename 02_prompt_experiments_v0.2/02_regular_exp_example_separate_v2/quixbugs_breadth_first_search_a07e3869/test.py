from node import Node
from breadth_first_search import breadth_first_search

def test__breadth_first_search():
    """The mutant changes 'while queue:' to 'while True:', which will lead to an IndexError."""
    # Create nodes where one is not reachable from the other
    start_node = Node(value='A', successors=[])
    goal_node = Node(value='B', successors=[])
    
    # This should return False, but the mutant will raise an error
    result = breadth_first_search(start_node, goal_node)
    assert result == False, "bfs should return False when goal is unreachable"