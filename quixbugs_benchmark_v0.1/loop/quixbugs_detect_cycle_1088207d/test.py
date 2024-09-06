from detect_cycle import detect_cycle
from node import Node 

def test__detect_cycle_mutant_killing():
    # Creating a non-cyclic list with even number of nodes (4 nodes)
    node4 = Node(value=4)
    node3 = Node(value=3, successor=node4)
    node2 = Node(value=2, successor=node3)
    node1 = Node(value=1, successor=node2)

    # Verify the correct implementation gives the expected result
    output_correct = detect_cycle(node1)
    assert output_correct is False, "The correct implementation should return False for non-cyclic list."

    # The mutant should raise an error when checking a non-cyclic list
    try:
        output_mutant = detect_cycle_mutant(node1)
    except Exception as e:
        print(f"Mutant error: {e}")  # Expecting an error here for the mutant

# Run the final test to see if it kills the mutant
test__detect_cycle_mutant_killing()