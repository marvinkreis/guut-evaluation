from reverse_linked_list import reverse_linked_list
from node import Node

def test__reverse_linked_list():
    # Create a linked list: 1 -> 2 -> 3 -> None
    head = Node(1, Node(2, Node(3)))

    # Reverse the linked list
    output = reverse_linked_list(head)

    # The new head should be 3
    assert output.value == 3, "The head of the reversed linked list should be 3"

    # Check if the second node is 2
    assert output.successor.value == 2, "The second node should be 2"

    # Check if the third node is 1, making sure the list is fully reversed
    assert output.successor.successor.value == 1, "The third node should be 1"

    # Check if the last node points to None
    assert output.successor.successor.successor is None, "The last node should point to None"