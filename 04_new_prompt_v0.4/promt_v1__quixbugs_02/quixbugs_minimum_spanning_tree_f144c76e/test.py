from minimum_spanning_tree import minimum_spanning_tree

def test__minimum_spanning_tree_killing_mutant():
    """
    Test the minimum_spanning_tree function with a specific edge weight input.
    The test checks that the correct minimum spanning tree is returned for the baseline,
    while the mutant will raise a RuntimeError due to wrong group updates.
    """
    weight_by_edge = {
        (1, 2): 10,
        (2, 3): 15,
        (3, 4): 10,
        (1, 4): 10
    }

    # Expected output for the baseline
    expected_output = {(1, 2), (3, 4), (1, 4)}
    
    # Run the function and check for the correct output on baseline
    output = minimum_spanning_tree(weight_by_edge)
    assert output == expected_output, f"Expected {expected_output}, got {output}"
    
    print(f"output = {output}")

# Run the test and observe behavior
test__minimum_spanning_tree_killing_mutant()