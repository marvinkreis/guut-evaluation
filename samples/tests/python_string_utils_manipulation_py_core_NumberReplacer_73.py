from string_utils.manipulation import __StringFormatter
import re


def test():
    formatter = __StringFormatter("")
    match = re.match(r"(a)(b)", "ab")

    assert formatter._StringFormatter__ensure_right_space_only(match) == "a "
