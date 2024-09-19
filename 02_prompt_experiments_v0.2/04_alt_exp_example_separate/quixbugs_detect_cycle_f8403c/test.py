from detect_cycle import detect_cycle
from node import Node

def test__detect_cycle():
    """Removing the check for hare being None in the detect_cycle function
       would cause it to throw an AttributeError on an empty list."""
    
    # Test case: empty list
    empty_list = None
    assert detect_cycle(empty_list) is False, "Expected false for an empty list"
    
    # Test case: single node with no successor
    single_node_list = Node(value=1)  # No successor
    assert detect_cycle(single_node_list) is False, "Expected false for a single node without a cycle"