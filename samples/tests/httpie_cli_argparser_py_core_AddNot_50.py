from httpie.cli.argparser import HTTPieArgumentParser
from httpie.context import Environment
from types import SimpleNamespace


def test():
    parser = HTTPieArgumentParser()
    try:
        parser.parse_args(Environment(),
            args=[],
            namespace=SimpleNamespace(debug=False,
            ignore_stdin=False,
            request_type="no",
            offline=False,
            download=True,
            download_resume=True,
            output_file=True,
            quiet=True,
            verbose=False,
            output_options={},
            output_options_history={},
            prettify="all",
            format_options=None,
            method=None,
            request_items=[],
            url="example.org",
            default_scheme="https",
            auth_type=None,
            auth=None,
            ignore_netrc=True,
            compress=False))
    except SystemExit:
        assert False

