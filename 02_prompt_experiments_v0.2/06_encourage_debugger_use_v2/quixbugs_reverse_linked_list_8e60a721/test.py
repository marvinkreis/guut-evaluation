from reverse_linked_list import reverse_linked_list
from node import Node

def test__reverse_linked_list():
    """The mutant fails to reverse the linked list and returns None instead of the correct reversed list."""
    
    # Creating a linked list: 1 -> 2 -> 3
    node3 = Node(3)
    node2 = Node(2, successor=node3)
    node1 = Node(1, successor=node2)
    
    # Test the correct implementation
    output = reverse_linked_list(node1)
    assert output is not None, "The output must not be None."
    
    # Verify the values in the reversed linked list
    values = []
    while output:
        values.append(output.value)
        output = output.successor

    # Assert that the correct output should be [3, 2, 1]
    assert values == [3, 2, 1], "The resulting linked list values must be [3, 2, 1]"