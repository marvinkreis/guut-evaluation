from topological_ordering import topological_ordering
from node import Node

def test__topological_ordering_kills_mutant():
    # Setting up a directed acyclic graph
    a = Node(value='A')  
    b = Node(value='B', incoming_nodes=[a])  
    c = Node(value='C', incoming_nodes=[a])  
    d = Node(value='D', incoming_nodes=[b, c])  
    e = Node(value='E', incoming_nodes=[c])  
    f = Node(value='F', incoming_nodes=[d])  

    # Defining outgoing nodes
    a.outgoing_nodes = [b, c]
    b.outgoing_nodes = [d]
    c.outgoing_nodes = [d, e]
    d.outgoing_nodes = [f]

    nodes = [a, b, c, d, e, f]

    # Run the correct topological ordering function
    output_correct = topological_ordering(nodes)

    # Creating the expected output
    expected_output = ['A', 'B', 'C', 'D', 'E', 'F']

    # Test assertion for correct implementation
    assert [node.value for node in output_correct] == expected_output, "The topological ordering is incorrect"

    # Simulating the mutant behavior
    def mutant_topological_ordering(nodes):
        ordered_nodes = [node for node in nodes if not node.incoming_nodes] # Initial node with no incoming edges

        for node in ordered_nodes:
            for nextnode in node.outgoing_nodes:
                if set(ordered_nodes).issuperset(nextnode.outgoing_nodes) and nextnode not in ordered_nodes:
                    ordered_nodes.append(nextnode)

        return ordered_nodes

    # Run the mutant function
    output_mutant = mutant_topological_ordering(nodes)

    # Assert that the mutant's output is different from the expected output
    assert [node.value for node in output_mutant] != expected_output, "The mutant should fail to produce the correct topological ordering"

# Execute the test
test__topological_ordering_kills_mutant()