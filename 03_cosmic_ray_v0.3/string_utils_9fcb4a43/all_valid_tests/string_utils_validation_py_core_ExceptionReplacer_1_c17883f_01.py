from string_utils.validation import is_isbn_10

def test__is_isbn_10_edge_case():
    """The mutant should be unable to correctly handle an ISBN-like input with mixed formats."""
    # The input is not a valid ISBN-10, as it includes invalid characters (hyphens)
    edge_case_isbn = '123-456-7890'  
    output = is_isbn_10(edge_case_isbn)
    
    assert output is False, "is_isbn_10 should return False for mixed format ISBN inputs"