"""Smoke test: load the folder the way ComfyUI loads a custom node and run it.

python tests/test_happyin_canny.py  -> exit 0 on success, 1 on failure.
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
# a sparse patch: one strong line (enough heat for the mask, far from hot) over faint texture that only
# the 5/20 thresholds would pick up
img[300:420, 450:700] += rng.normal(0, 0.03, (120, 250, 1)).astype(np.float32)
img[310:316, 450:700] = 0.95   # measured: heat peaks at 139 here, so mask yes (>30), hot zone no (<150)
o = run(img)
mask = o["detail_filter_mask"][0, ..., 0].numpy() > 0.5
assert mask[60:180, 60:180].mean() > 0.6, "busy square must be in the detail mask"
assert mask[440:, :].sum() == 0 and mask[:, 720:].sum() == 0, "flat area must stay out of the mask"
assert mask[306:320, 470:680].mean() > 0.5, "the single strong line must be inside the detail mask"
assert o["combined_edges_final"].sum() > 0
# the fine hot-zone pass fires inside the busy square and nowhere on the flat area
hot = o["extra_detailed_edges_pass3"][0, ..., 0].numpy() > 0.5
assert hot[40:200, 40:200].sum() > 0, "hot-zone pass must find fine edges in the busy square"
assert hot[220:, 220:].sum() == 0, "hot-zone pass must stay off the sparse patch and the flat area"

# negative control: a flat frame has no detail anywhere
flat = run(np.full((256, 256, 3), 0.5, np.float32))
assert all(float(v.sum()) == 0 for k, v in flat.items() if k != "gradient_heatmap"), "flat frame must give empty masks"

# other IMAGE tensors ComfyUI produces: RGBA, one channel, fp16, values past [0, 1], batch of 2
ref = o["combined_edges_final"]
rgba = np.concatenate([img, np.ones_like(img[..., :1])], axis=-1)
for label, t in (("rgba", torch.from_numpy(rgba)[None]),
                 ("fp16", torch.from_numpy(img)[None].half()),
                 ("bf16", torch.from_numpy(img)[None].bfloat16())):   # the original node crashed here
    out = node.apply_canny_logic(t)
    assert out[0].shape == (1, 512, 768, 3), label
    if label == "rgba":
        assert torch.equal(out[0], ref), "RGBA input must give the same edges as RGB"
    else:   # half precision rounds a few pixels to the next 8-bit level, so a few edges may move
        assert (out[0] != ref).float().mean() < 0.01, f"{label} input must give nearly the same edges as RGB"
# values past [0, 1] are clipped, not wrapped round uint8 (the original turned 1.15 into 37)
over = torch.from_numpy(img * 1.3 - 0.15)[None]
assert torch.equal(node.apply_canny_logic(over)[0], node.apply_canny_logic(over.clamp(0, 1))[0])
gray = node.apply_canny_logic(torch.from_numpy(img[..., :1].copy())[None])
gray3 = node.apply_canny_logic(torch.from_numpy(np.repeat(img[..., :1], 3, axis=-1))[None])
assert gray[0].shape == (1, 512, 768, 3)
assert torch.equal(gray[0], gray3[0]), "one channel must behave as the same grey in RGB"
batch = node.apply_canny_logic(torch.from_numpy(np.stack([img, img])))
assert batch[0].shape[0] == 2 and torch.equal(batch[0][0], batch[0][1])

print("PASS: Happyin_Canny loads and finds the busy region")
