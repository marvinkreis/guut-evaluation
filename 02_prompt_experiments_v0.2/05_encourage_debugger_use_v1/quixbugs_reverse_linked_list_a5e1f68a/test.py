from node import Node
from reverse_linked_list import reverse_linked_list

def test__reverse_linked_list():
    """The mutant fails to return a proper Node instance when reversing a linked list."""
    # Creating a simple linked list: 1 -> 2 -> 3
    node3 = Node(value=3)
    node2 = Node(value=2, successor=node3)
    node1 = Node(value=1, successor=node2)

    output = reverse_linked_list(node1)
    
    # The output should be a valid node (head of the reversed list)
    assert isinstance(output, Node), "Expected output should be a Node instance"
    
    # Checking if the output is indeed a reversed list
    assert output.value == 3, "The new head value should be 3"
    assert output.successor.value == 2, "The next value should be 2"
    assert output.successor.successor.value == 1, "The last value should be 1"
    assert output.successor.successor.successor == None, "The last node should point to None"