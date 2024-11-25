from io import StringIO
from types import SimpleNamespace

import httpie.downloads as downloads


class MockInt(int):
    def __bool__(self):
        return True

def test():
    output = StringIO()
    prt = downloads.ProgressReporterThread(SimpleNamespace(
        downloaded=1,
        total_size=MockInt(0),
    ), output)

    prt._prev_time -= prt._update_interval
    prt.report_speed()

    assert "0.00 %" in output.getvalue()
