# Test to verify that the HTML_RE regex compiles correctly, which will fail in the mutant due to syntax error.
def test__html_re_compile_validation():
    """
    Test if the HTML_RE regex compiles without error. The baseline should compile successfully, whereas
    the mutant introduces a division operator instead of a logical OR, causing a syntax error.
    """
    try:
        import string_utils._regex as regex_module
        regex_module.HTML_RE
    except SyntaxError:
        assert False, "HTML_RE regex failed to compile due to SyntaxError"
    except Exception:
        assert False, "HTML_RE regex failed to compile due to another error"
    
    assert True  # Compiled successfully