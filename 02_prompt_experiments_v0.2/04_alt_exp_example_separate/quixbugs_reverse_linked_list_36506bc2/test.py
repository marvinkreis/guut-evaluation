from reverse_linked_list import reverse_linked_list
from node import Node

def test__reverse_linked_list():
    """Removing 'prevnode = node' causes the function to fail and produce an empty list upon reversal."""
    # Create a simple linked list: 1 -> 2 -> 3
    node1 = Node(1)
    node2 = Node(2)
    node3 = Node(3)
    node1.successor = node2
    node2.successor = node3

    output = reverse_linked_list(node1)
    
    # Collecting output values for verification
    output_values = []
    current = output
    while current:
        output_values.append(current.value)
        current = current.successor
    
    assert output_values == [3, 2, 1], "List must be reversed properly!"
    assert len(output_values) > 0, "The output should not be an empty list."