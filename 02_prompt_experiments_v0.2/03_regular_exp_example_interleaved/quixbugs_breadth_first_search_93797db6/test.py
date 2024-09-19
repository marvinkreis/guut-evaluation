from breadth_first_search import breadth_first_search
from node import Node

def test__breadth_first_search():
    """The mutant incorrectly handles empty queues in BFS. It should return False if the goal node is unreachable."""
    # Create a simple graph
    node_a = Node(value='A', successors=[])
    node_b = Node(value='B', successors=[])
    node_a.successors = [node_b]  # A -> B
    
    goal_node = Node(value='C', successors=[])  # Node C is unreachable

    result = breadth_first_search(node_a, goal_node)
    assert result == False, "BFS should return False when the goal node is unreachable"