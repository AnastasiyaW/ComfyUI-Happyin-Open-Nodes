"""Smoke test: load the folder the way ComfyUI loads a custom node and run it.

python tests/test_node.py  -> exit 0 on success, 1 on failure.
Needs torch, numpy, opencv (all present in a ComfyUI environment).
"""
import importlib.util, pathlib, sys
import numpy as np, torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("comfyui_happyin_canny", ROOT / "__init__.py",
                                              submodule_search_locations=[str(ROOT)])
pkg = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pkg
spec.loader.exec_module(pkg)

node = pkg.NODE_CLASS_MAPPINGS["Happyin_Canny"]()
assert pkg.NODE_DISPLAY_NAME_MAPPINGS["Happyin_Canny"] == "Happyin Canny"
names = node.RETURN_NAMES


def run(img):
    outs = node.apply_canny_logic(torch.from_numpy(img)[None])
    assert len(outs) == 6 and all(o.shape == (1, *img.shape) for o in outs)
    return dict(zip(names, outs))


# flat grey frame with one busy 160x160 square (random checker) in the corner
rng = np.random.default_rng(0)
img = np.full((512, 768, 3), 0.5, np.float32)
img[40:200, 40:200] = np.kron(rng.integers(0, 2, (20, 20)), np.ones((8, 8)))[..., None]
o = run(img)
mask = o["detail_filter_mask"][0, ..., 0].numpy() > 0.5
assert mask[60:180, 60:180].mean() > 0.6, "busy square must be in the detail mask"
assert mask[300:, 400:].sum() == 0, "flat area must stay out of the mask"
assert o["combined_edges_final"].sum() > 0

# negative control: a flat frame has no detail anywhere
flat = run(np.full((256, 256, 3), 0.5, np.float32))
assert all(float(v.sum()) == 0 for k, v in flat.items() if k != "gradient_heatmap"), "flat frame must give empty masks"

print("PASS: Happyin_Canny loads and finds the busy region")
