# Root conftest so pytest puts the repo root on sys.path, which lets tests import the
# top-level `eval` package (the evaluation harness) without installing anything extra.
# The `embr` package itself is installed editable, so it never needed this.
