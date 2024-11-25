from unittest.mock import MagicMock, patch
from flake8.checker import Manager
from flake8.main.options import JobsArgument



# adapted from flake8's test suite
def test_make_checkers():
    style_guide = MagicMock(**{
        'options.diff': False,
        'options.jobs': JobsArgument("4"),
        'options.filename': ["a"],
        'options._running_from_vcs': True,
    })
    files = ['file1', 'file2']
    checkplugins = MagicMock()
    checkplugins.to_dictionary.return_value = {
        'ast_plugins': [],
        'logical_line_plugins': [],
        'physical_line_plugins': [],
    }
    with patch('flake8.checker.multiprocessing', None):
        manager = Manager(style_guide, files, checkplugins)

    with patch('flake8.utils.filenames_from') as filenames_from:
        filenames_from.side_effect = [['#'], ['file1']]
        with patch('flake8.processor.FileProcessor'):
            manager.make_checkers()

    assert not manager._all_checkers
