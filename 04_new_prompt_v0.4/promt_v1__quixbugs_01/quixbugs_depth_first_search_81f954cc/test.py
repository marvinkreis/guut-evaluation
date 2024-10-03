from depth_first_search import depth_first_search
from node import Node

def test__depth_first_search_mutant_killing():
    """
    Test situation where depth_first_search does not terminate due to cycles.
    The input graph contains a loop (A -> B -> B; B -> C), which causes the mutant
    to recurse indefinitely while the baseline returns True. This should confirm 
    that the mutant is incorrect as it fails to mark nodes as visited.
    """
    A = Node(value='A', successors=[])
    B = Node(value='B', successors=[])
    C = Node(value='C', successors=[])

    # Construct a graph with a cycle
    A.successors.append(B)
    B.successors.append(B)  # Self-loop on B
    B.successors.append(C)   # Path to C

    # Expect True from the baseline, but the mutant will run indefinitely.
    depth_first_search(A, C)  # Not asserting True to force the issue on the mutant.