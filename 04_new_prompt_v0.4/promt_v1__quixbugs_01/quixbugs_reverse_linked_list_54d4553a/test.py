from reverse_linked_list import reverse_linked_list
from node import Node

def test_reverse_linked_list_killing_mutant():
    """
    Test the reverse_linked_list function for multi-node and single-node linked lists. 
    The multi-node list (1 -> 2 -> 3) should return (3 -> 2 -> 1), and the single-node (1) should retain its value (1). 
    The mutant should produce an empty list for the multi-node case and None for the single-node case, demonstrating its failure.
    """
    
    # Multi-node linked list: 1 -> 2 -> 3
    node3 = Node(value=3)
    node2 = Node(value=2, successor=node3)
    node1 = Node(value=1, successor=node2)
    
    # Reverse the linked list
    new_head_multi = reverse_linked_list(node1)

    # Collect values of the reversed multi-node linked list
    multi_reversed_values = []
    current_multi = new_head_multi
    while current_multi:
        multi_reversed_values.append(current_multi.value)
        current_multi = current_multi.successor

    # Assert that the output matches the expected reversed list
    assert multi_reversed_values == [3, 2, 1], f"Expected [3, 2, 1] but got {multi_reversed_values}"
    
    # Single-node linked list: 1
    single_node = Node(value=1)

    # Reverse the single-node linked list
    new_head_single = reverse_linked_list(single_node)

    # Collect value of the reversed single-node linked list
    single_reversed_value = new_head_single.value if new_head_single else None

    # Assert that the output matches the expected value
    assert single_reversed_value == 1, f"Expected 1 but got {single_reversed_value}"