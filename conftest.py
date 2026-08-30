# Root conftest so pytest puts the repo root on sys.path, which lets tests import the
# top-level `eval` package (the evaluation harness) without installing anything extra.
# The `embr` package itself is installed editable, so it never needed this.

#: Files whose gated tests load the real Ouro checkpoint through torch's memory-mapped
#: reader. On Windows that load hits an access violation when it happens late in the
#: suite, after matplotlib and sentence-transformers have fragmented the address space;
#: the identical load succeeds in a fresh process. Running these files first gives the
#: checkpoint the clean address space it needs, and costs nothing when the gate skips.
_LOADS_REAL_WEIGHTS = ("test_model.py", "test_context_attribution.py")


def pytest_collection_modifyitems(config, items):
    items.sort(key=lambda item: 0 if item.path.name in _LOADS_REAL_WEIGHTS else 1)
