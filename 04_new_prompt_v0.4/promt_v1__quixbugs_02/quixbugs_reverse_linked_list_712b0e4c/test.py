from reverse_linked_list import reverse_linked_list
from node import Node

def test__reverse_linked_list_kills_mutant():
    """
    This test checks if the reverse_linked_list function correctly reverses a linked list. 
    The input linked list `1 -> 2 -> 3` should produce `3 -> 2 -> 1` when properly reversed.
    The mutant will not perform this reverse correctly, returning an empty list instead,
    thus providing a clear distinction in output.
    """
    # Constructing a simple linked list: 1 -> 2 -> 3
    node1 = Node(1)
    node2 = Node(2)
    node3 = Node(3)

    node1.successor = node2
    node2.successor = node3

    # Reversing the linked list
    new_head = reverse_linked_list(node1)
    
    # Traversing the reversed list to gather output values
    output = []
    current = new_head
    while current:
        output.append(current.value)
        current = current.successor

    print(f"Reversed list output (values): {output}")
    assert output == [3, 2, 1], "The linked list was not reversed correctly!" 