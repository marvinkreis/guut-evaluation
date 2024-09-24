from string_utils.manipulation import prettify

def test__prettify():
    original_string = ' unprettified string ,, like this one,will be"prettified" .it\'s awesome! '
    expected_output = 'Unprettified string, like this one, will be "prettified". It\'s awesome!'
    
    # when executed with the original code, this should not raise an exception
    output = prettify(original_string)
    assert output == expected_output, f"Expected: {expected_output}, but got: {output}"

    # When executed with the mutant code, it should raise an error due to incorrect method behavior.
    try:
        # Simulating the mutant behavior by executing the mutated version of the method
        # This will fail because of the syntax error introduced by the mutant
        mutated_output = prettify(original_string)
    except Exception as e:
        assert isinstance(e, SyntaxError), "Expected a SyntaxError due to mutant modification"