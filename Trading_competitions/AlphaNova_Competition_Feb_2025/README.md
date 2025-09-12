How to install dependenies and run the code.

We are using `uv` as a package management tool. Feel free to use any other package manager that understands `pyproject.toml` file.
1. Install uv `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. `uv venv`
3. `source .venv/bin/activate`
4. `uv sync`
5. `jupyter notebook AlphaNovaContestTemplate.24.02.25.ipynb`

Alternatively you can use `pip`:
1. `python3 -m venv alphanova`
2. `source alphanova/bin/activate`
3. `pip install -r requirements.txt`
4. `jupyter notebook AlphaNovaContestTemplate.24.02.25.ipynb`