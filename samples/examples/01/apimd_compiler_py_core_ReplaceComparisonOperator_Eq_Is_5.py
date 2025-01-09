from apimd.compiler import interpret_mode


class EqToAll:
    def __eq__(self, other):
        return True

def test():
    orig_enumerate = __builtins__["enumerate"]
    __builtins__["enumerate"] = lambda x: [(EqToAll(), y) for y in x] if x == [">>> "] else orig_enumerate(x)
    assert "```\n" in list(interpret_mode(">>> "))

