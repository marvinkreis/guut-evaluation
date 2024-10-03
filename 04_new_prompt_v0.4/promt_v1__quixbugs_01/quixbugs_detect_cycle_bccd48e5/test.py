from detect_cycle import detect_cycle
from node import Node

def test__detect_cycle_empty_list():
    """
    Test detecting a cycle in an empty linked list. The input is None which represents
    the head of an empty list. The baseline should return False while the mutant will raise
    an AttributeError due to the design change that omits the None check.
    """
    output = detect_cycle(None)  # Empty linked list (no nodes)
    assert output is False  # This should pass in the baseline implementation