from reverse_linked_list import reverse_linked_list
from node import Node

def test__reverse_linked_list():
    """Removing 'prevnode = node' causes the function to return None 
    instead of the new head of the reversed linked list."""
    
    # Create a linked list: 1 -> 2 -> 3
    head = Node(1, Node(2, Node(3)))
    
    # Call the reverse function
    reversed_head = reverse_linked_list(head)
    
    # Verifying expressions
    assert reversed_head is not None, "reversed_head must not be None"
    assert reversed_head.value == 3, "expected head value to be 3"
    assert reversed_head.successor.value == 2, "expected first successor to be 2"