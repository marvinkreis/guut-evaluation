from detect_cycle import detect_cycle
from node import Node

def test__detect_cycle():
    # Create a non-cyclic linked list: 1 -> 2 -> 3 -> None
    node3 = Node(3)
    node2 = Node(2, node3)
    node1 = Node(1, node2)

    # Test case for non-cyclic list
    assert detect_cycle(node1) == False, "Failed to detect non-cyclic list"

    # Create a cyclic linked list: 1 -> 2 -> 3 -> 1 (cycle back to node 1)
    node1.successor = node2
    node2.successor = node3
    node3.successor = node1  # Creating cycle

    # Test case for cyclic list
    assert detect_cycle(node1) == True, "Failed to detect cyclic list"

    # Now let's add one more node to the linked list and test the mutant's behavior
    # Extend list with an additional node without a cycle
    node4 = Node(4)        # New node
    node3.successor = node4  # Non-cyclic end
    node4.successor = None  # This breaks the cycle
    
    # At this point, we have a linear list: 1 -> 2 -> 3 -> 4 -> None
    # The hare will traverse 2 steps ahead and should detect the end
    assert detect_cycle(node1) == False, "Failed to detect a non-cyclic list in extended version"