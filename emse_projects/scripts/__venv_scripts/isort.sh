uv venv --allow-existing --relocatable --python cpython-3.8.19-linux-x86_64-gnu
uv pip install poetry
source .venv/bin/activate
sed -i '/deprecated_finder/d' ./pyproject.toml
poetry export --without-hashes --format=requirements.txt > requirements.txt
uv pip install -r requirements.txt coverage pytest pytest-cov
