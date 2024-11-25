from types import SimpleNamespace

import flake8.main.git as git


def test():
    mock_process = SimpleNamespace(communicate=lambda: ("stdout", "stderr"), returncode=-1)
    git.__dict__["piped_process"] = lambda x: mock_process

    assert git.find_git_directory() is None
