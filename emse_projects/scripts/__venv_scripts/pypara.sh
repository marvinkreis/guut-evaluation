uv venv --allow-existing --relocatable --python cpython-3.8.19-linux-x86_64-gnu
uv pip install setuptools
source .venv/bin/activate
python << EOF
from distutils.core import run_setup
from pathlib import Path

setup = run_setup("setup.py")
reqs = "\n".join(setup.install_requires)
Path("./requirements.txt").write_text(reqs)
EOF
uv pip install -r requirements.txt coverage pytest pytest-cov
