from kheapsort import kheapsort

def test__kheapsort():
    """The mutant should yield a list with unique elements, but it incorrectly yields duplicates."""
    output = list(kheapsort([10, 20, 5, 4, 15], 2))
    assert len(output) == len(set(output)), "Output must have unique elements."