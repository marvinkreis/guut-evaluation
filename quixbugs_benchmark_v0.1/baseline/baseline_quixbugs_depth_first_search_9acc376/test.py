from depth_first_search import depth_first_search
from node import Node

def test_depth_first_search():
    # Create nodes
    node_a = Node(value='A')
    node_b = Node(value='B')
    node_c = Node(value='C')
    
    # Set up a cycle in the graph (A -> B -> A)
    node_a.successors = [node_b]
    node_b.successors = [node_a]  # creates a cycle

    # Test that the goal node is reachable; since there is a cycle, should still terminate correctly.
    assert depth_first_search(node_a, node_a) == True, "Should find node A from A (self loop)"

    # Introduce an additional node to allow a valid path
    node_d = Node(value='D')
    node_b.successors.append(node_c)  # B -> C

    # Now, test findability of C through A
    assert depth_first_search(node_a, node_c) == True, "Should find node C from A via B"

    # Verify that node C is reachable from A while ensuring we have a proper path without infinite recursion
    # If nodesvisited.add(node) is commented out in the mutant, this would lead to infinite recursion
    node_e = Node(value='E')
    node_b.successors.append(node_e)  # Additional path to ensure multiple steps

    assert depth_first_search(node_a, node_e) == True, "Should find node E from A via B"

    # Test finding a non-reachable node
    assert depth_first_search(node_a, node_d) == False, "Should not find D from A"

    # Since all cases that include cycles should terminate, we expect this to pass with the correct code