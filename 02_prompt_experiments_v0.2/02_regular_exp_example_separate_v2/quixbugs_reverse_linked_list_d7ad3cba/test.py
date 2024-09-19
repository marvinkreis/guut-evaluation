from node import Node
from reverse_linked_list import reverse_linked_list

def test__reverse_linked_list():
    """Removing the assignment to prevnode causes the function to always return an empty list."""
    
    # Create a linked list: 1 -> 2 -> 3
    node1 = Node(value=1)
    node2 = Node(value=2)
    node3 = Node(value=3)
    node1.successor = node2
    node2.successor = node3
    
    # Run the function
    reversed_head = reverse_linked_list(node1)
    
    # Collect the reversed list values
    reversed_values = []
    current = reversed_head
    while current:
        reversed_values.append(current.value)
        current = current.successor
    
    # Assert the output should not be empty
    assert len(reversed_values) > 0, "The reversed linked list must not be empty"