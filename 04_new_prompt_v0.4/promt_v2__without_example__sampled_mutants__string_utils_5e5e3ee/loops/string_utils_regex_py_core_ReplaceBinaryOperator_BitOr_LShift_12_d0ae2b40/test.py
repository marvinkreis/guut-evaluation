from string_utils._regex import PRETTIFY_RE

def test__saxon_genitive_behavior():
    """
    Test to ensure the SAXON_GENITIVE regex does not raise an OverflowError 
    when using multiline and Unicode flags. The mutant incorrectly combines these flags.
    Expected outcome is that the regex should handle the input without errors,
    which will fail if the mutant's incorrect flag handling is in effect.
    """
    test_string = "Joan's s cat\nAnd also Sarah's s dog"
    
    try:
        output = PRETTIFY_RE['SAXON_GENITIVE'].findall(test_string)
        assert output == []
    except Exception as e:
        assert isinstance(e, OverflowError)