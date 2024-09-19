from reverse_linked_list import reverse_linked_list
from node import Node

def test_reverse_linked_list():
    # Create a linked list: 1 -> 2 -> 3
    node3 = Node(3)
    node2 = Node(2, node3)
    node1 = Node(1, node2)

    # Reverse the linked list
    new_head = reverse_linked_list(node1)

    # Check the new head
    assert new_head.value == 3
    assert new_head.successor.value == 2
    assert new_head.successor.successor.value == 1
    assert new_head.successor.successor.successor is None

    # Verify the original head (node1) now points to None
    assert node1.successor is None