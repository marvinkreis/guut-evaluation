from string_utils.validation import is_ip_v6

def test__is_ip_v6_mutant_killer():
    """
    Test specifically an invalid IPv6 address format that the mutant will incorrectly accept as valid.
    The input '2001::db8:85a3:0000:0000:8a2e:370:xyz' should return False in the baseline but True in the mutant,
    thus killing the mutant.
    """
    output = is_ip_v6('2001::db8:85a3:0000:0000:8a2e:370:xyz')
    assert output == False, f"Expected False, got {output}"