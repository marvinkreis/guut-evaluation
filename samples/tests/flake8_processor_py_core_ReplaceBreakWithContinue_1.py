from flake8.processor import FileProcessor
import argparse
import tokenize


# from flake8 test suite
def options_from(**kwargs):
    """Generate a Values instances with our kwargs."""
    kwargs.setdefault('hang_closing', True)
    kwargs.setdefault('max_line_length', 79)
    kwargs.setdefault('max_doc_length', None)
    kwargs.setdefault('indent_size', 4)
    kwargs.setdefault('verbose', False)
    kwargs.setdefault('stdin_display_name', 'stdin')
    kwargs.setdefault('disable_noqa', False)
    return argparse.Namespace(**kwargs)

def test():
    p = FileProcessor("filename.py", options_from(), lines=["import sys\n"] * 10)
    p._file_tokens = [
            (tokenize.ENDMARKER, None, (2, None), (2, None), None),
            (tokenize.NL, None, (2, None), (2, None), None),
    ]
    p.noqa_line_for(0)
    assert not p._noqa_line_mapping


