from flake8.style_guide import find_noqa
import flake8.defaults as defaults
from types import SimpleNamespace

i = 0

def test():
    def mock_search(text):
        global i
        i += 1
        return None

    defaults.NOQA_INLINE_REGEXP = SimpleNamespace(search=mock_search)

    for j in range(2):
        for k in range(513):
            find_noqa(f"# noqa: E{k}")
    assert i > 513
