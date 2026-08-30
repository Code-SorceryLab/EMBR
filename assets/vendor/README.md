# Vendored third-party code

Nothing in this directory is ours, and nothing in it is a research dependency. The eval, the
scorer and every number in the paper run with `dependencies = []`. This exists only so the
3D page can be opened from a `file://` path with no network, which is the same constraint the
2D demo already meets and which a CDN link would quietly break.

| file | what | version | licence |
|---|---|---|---|
| `three.min.js` | [three.js](https://threejs.org), WebGL renderer | **r149** | MIT, Copyright 2010-2023 Three.js Authors |

Provenance, so the bytes are checkable rather than trusted:

```
url     https://unpkg.com/three@0.149.0/build/three.min.js
size    608,081 bytes
sha256  8a5f7249903b54d30f79f708699d2fed2d6a1d0741a4cd41377d1f01bb5a2271
```

Verify with `python -c "import hashlib,pathlib;print(hashlib.sha256(pathlib.Path('assets/vendor/three.min.js').read_bytes()).hexdigest())"`,
which is also what `tests/test_build_brain3d.py` does on every run.

**Why r149 and not the current release.** From r150 onward three.js stopped shipping a UMD
build, and its ES module build imports a second file (`three.core.js`). Two files cannot be
inlined into one self-contained page without an import map and a fetch, and a fetch is exactly
what the page is not allowed to do. r149 is the last release that is one file with no imports.
It is a renderer for a point cloud and some lines, so nothing newer is needed.

**Not committed to the paper.** No figure, table or statistic depends on this file. Deleting
it breaks the 3D page and nothing else.
