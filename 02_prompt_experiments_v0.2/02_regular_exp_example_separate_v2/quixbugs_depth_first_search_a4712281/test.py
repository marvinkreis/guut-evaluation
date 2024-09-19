from node import Node
from depth_first_search import depth_first_search

def test__depth_first_search():
    """The mutant should lead to an infinite loop due to lack of node visitation tracking."""
    # Setup a graph where A cannot reach C
    A = Node(value='A')
    B = Node(value='B')
    C = Node(value='C')

    A.successors.append(B)  # A -> B
    B.successors.append(A)  # B -> A (cycle between A and B)
    # C is disconnected from A and B

    output = depth_first_search(A, C)
    assert output is False, "depth_first_search must return False when goal is unreachable"