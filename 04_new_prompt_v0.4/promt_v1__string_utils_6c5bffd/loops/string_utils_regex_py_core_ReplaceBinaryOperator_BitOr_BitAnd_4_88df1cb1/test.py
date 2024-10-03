from string_utils._regex import HTML_RE

def test__html_re():
    """
    Test HTML_RE regex against a complex multi-line HTML string with irregular formatting. 
    The Baseline should correctly capture all elements with whitespace included, 
    while the Mutant may fail to return the complete structure.
    """
    html_input = """
    <div>   
        <p> This  is a   paragraph. </p>   
        <ul>
              <li>Item 1</li>
               <li>Item 2</li>
        </ul>
       <!-- Comment -->
    </div>
    """
    output = HTML_RE.findall(html_input)
    assert any("<div>   " in match[0] for match in output)  # Check if the output contains whitespace in the div element
    assert "<!-- Comment -->" in "".join(match[0] for match in output)  # Expecting comment to be present
    assert len(output) == 4  # Expecting number of matched elements to be 4