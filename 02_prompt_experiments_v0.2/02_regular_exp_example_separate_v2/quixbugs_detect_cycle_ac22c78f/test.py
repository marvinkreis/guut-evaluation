from detect_cycle import detect_cycle
from node import Node  # Ensure Node class is properly imported

def test__detect_cycle():
    """Test detect_cycle function with None and non-cyclical input."""
    
    # Test with None input
    output = detect_cycle(None)
    assert output is False, "detect_cycle should return False for None input"
    
    # Create a simple linked list with 3 nodes without a cycle
    node1 = Node(value=1)
    node2 = Node(value=2)
    node3 = Node(value=3)
    
    node1.successor = node2
    node2.successor = node3
    node3.successor = None  # Making it a non-cyclical linked list

    output_noncyclic = detect_cycle(node1)
    assert output_noncyclic is False, "detect_cycle should return False for non-cyclical linked list"

# Call the test function directly to execute the testing mechanism
test__detect_cycle()