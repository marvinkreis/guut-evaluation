from breadth_first_search import breadth_first_search
from node import Node

def test__breadth_first_search_detects_mutant():
    """
    This test checks the breadth_first_search function's handling of a goal node.
    The baseline is expected to return False, while the mutant should fail.
    By asserting we can also ensure we have clear detection of the mutant.
    """
    # Create nodes
    node_a = Node(value='A')
    node_b = Node(value='B')
    node_c = Node(value='C')

    # Set up the graph structure
    node_a.successors = [node_b]  # A -> B
    node_b.successors = []         # B does not lead to C
    # Node C is isolated and not reachable via A

    # Expect that A cannot reach C, thus the output should be False
    output = breadth_first_search(node_a, node_c)
    print(f"Output for searching from A to C: {output}")  # Expected to be False

    # To kill the mutant, we will check this condition
    if output is not False:
        raise AssertionError("Mutant detected: the output is not False when it should be.")