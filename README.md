# Crop Classification Portfolio Project

- `crop_classification/` package with reusable geospatial modules
- command-line workflow in `main.py`
- STAC search and download example in `scripts/download_s2.py`
- package metadata with `pyproject.toml`
- documentation in `docs/index.md`
- basic unit tests in `tests/`
- CI configuration in `.github/workflows/python-package.yml`


## Quick check

Without dataset or credentials, you can still verify:

```bash
source .venv/bin/activate
pip install -r requirements.txt
python main.py --help
python main.py search --help
```

## Setup (basic)

1. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy the config template:
   ```bash
   cp crop_classification/config_template.py config.py
   ```
4. Add your Azure and Copernicus credentials to `config.py`.

## Notes

- `config.py` is intentionally excluded from version control.
- Actual Sentinel-2 assets are not included in the repo.

## License

MIT License
