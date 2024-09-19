from reverse_linked_list import reverse_linked_list
from node import Node

def test__reverse_linked_list():
    """The mutant lacks the line that updates 'prevnode' while reversing the list,
    causing it to incorrectly return the original head instead of the new head."""

    def create_linked_list(values):
        head = None
        for value in reversed(values):
            head = Node(value=value, successor=head)
        return head

    def print_linked_list(head):
        values = []
        while head:
            values.append(head.value)
            head = head.successor
        return values

    # Create and reverse a linked list: 1 -> 2 -> 3 -> 4
    linked_list_head = create_linked_list([1, 2, 3, 4])
    correct_output = reverse_linked_list(linked_list_head)  # should return [4, 3, 2, 1]
    mutant_output = reverse_linked_list(linked_list_head)    # should return [1]

    print(f"Correct output: {print_linked_list(correct_output)}")
    print(f"Mutant output: {print_linked_list(mutant_output)}")
    
    # Verify that the outputs differ
    assert print_linked_list(correct_output) != print_linked_list(mutant_output), \
        "The outputs from the correct implementation and mutant should differ!"

# Run the test
test__reverse_linked_list()