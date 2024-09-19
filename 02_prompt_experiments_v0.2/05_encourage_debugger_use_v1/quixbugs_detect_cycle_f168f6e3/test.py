from detect_cycle import detect_cycle
from node import Node

def test__detect_cycle():
    """The mutant fails because it doesn't check if hare is None before checking hare.successor."""
    
    # Create a single-node linked list (non-cyclic)
    node = Node(value=1)
    
    # Check that the detection should return False
    result = detect_cycle(node)
    assert result == False, "Expected False for a single node linked list, but got True"

    # Create two nodes without cycle
    first_node = Node(value=1)
    second_node = Node(value=2)
    first_node.successor = second_node  # Link first to second
    
    # Check that the detection should return False for non-cyclic two-node list
    result_no_cycle = detect_cycle(first_node)
    assert result_no_cycle == False, "Expected False for a two-node non-cyclic linked list, but got True"

# Now we would run the test
test__detect_cycle()