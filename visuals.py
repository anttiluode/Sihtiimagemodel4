"""Pictures: a side-by-side comparison grid and a GIF of one image settling and being pinged."""
import numpy as np
from PIL import Image, ImageDraw
from sihti4 import gaussian_fill, block_mean_matrix

SCALE = 4


def _tile(a, frame=None):
    a = np.clip(np.asarray(a, float), 0, 1)
    rgb = np.repeat((a * 255).astype(np.uint8)[..., None], 3, -1)
    if frame is not None:                       # tint the hidden region's border
        from scipy.ndimage import binary_dilation
        edge = binary_dilation(~frame) & frame
        rgb[edge] = (rgb[edge] * .4 + np.array([230, 120, 40]) * .6).astype(np.uint8)
    im = Image.fromarray(rgb).resize((32 * SCALE, 32 * SCALE), Image.NEAREST)
    return im


def comparison(images, keys, labels, path, rows=10):
    """images: list of (target, obs, mask, outputs-dict, ...)."""
    rows = min(rows, len(images))
    W, H = 32 * SCALE + 6, 32 * SCALE + 6
    top = 34
    sheet = Image.new("RGB", (W * (len(keys) + 2), top + H * rows), (24, 24, 28))
    d = ImageDraw.Draw(sheet)
    for j, lab in enumerate(["truth", "observed"] + labels):
        d.text((j * W + 6, 10), lab, fill=(230, 230, 230))
    for i in range(rows):
        target, obs, mask, out = images[i][:4]
        tiles = [target, obs] + [out[k] for k in keys]
        for j, t in enumerate(tiles):
            sheet.paste(_tile(t, mask if j != 0 else None), (j * W + 3, top + i * H + 3))
    sheet.save(path)


def coarse_view(x, block=4):
    D = block_mean_matrix(32, block)
    g = 32 // block
    return np.kron((D @ x.ravel()).reshape(g, g), np.ones((block, block)))


def state_view(model, prep, states):
    """Colour each hidden pixel by the state of the patch centred nearest to it."""
    K = model.K
    rng = np.random.default_rng(3)
    palette = rng.uniform(.25, 1., (K, 3))
    img = np.full((32, 32, 3), .1)
    c = model.patch // 2
    owner = {}
    for p, row in enumerate(prep["pidx"]):
        owner.setdefault(row[c * model.patch + c], p)
    hidden = prep["hidden"]
    for h in hidden:
        y, x = divmod(h, 32)
        best = None
        for (py, px) in ((y, x), (y - 1, x), (y, x - 1), (y - 1, x - 1), (y + 1, x), (y, x + 1)):
            if 0 <= py < 32 and 0 <= px < 32 and py * 32 + px in owner:
                best = owner[py * 32 + px]
                break
        if best is not None:
            img[y, x] = palette[states[best]]
    return img


def _rgb_tile(a):
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((32 * SCALE, 32 * SCALE), Image.NEAREST)


def loop_gif(model, target, obs, mask, path, n_dirs=3, size=2.5):
    """Settle, then ping along each uncertain layout direction and settle again."""
    prep = model.prepare(obs, mask)
    base = model.settle(prep, gaussian_fill(obs, mask), record=True)
    frames = []
    W = 32 * SCALE + 6

    def panel(x, states, caption, energy):
        im = Image.new("RGB", (W * 4, 32 * SCALE + 40), (24, 24, 28))
        d = ImageDraw.Draw(im)
        for j, (t, lab) in enumerate([(target, "truth"), (x, "image"),
                                      (coarse_view(x, model.block), "coarse layout (4x4 means)")]):
            im.paste(_tile(t, mask if j == 1 else None), (j * W + 3, 3))
            d.text((j * W + 6, 32 * SCALE + 8), lab, fill=(220, 220, 220))
        im.paste(_rgb_tile(state_view(model, prep, states)), (3 * W + 3, 3))
        d.text((3 * W + 6, 32 * SCALE + 8), "local states in the hole", fill=(220, 220, 220))
        d.text((6, 32 * SCALE + 24), f"{caption}   E = {energy:.1f}", fill=(255, 190, 90))
        return im

    for i, ((x, st), e) in enumerate(zip(base["frames"], base["energies"])):
        frames.append(panel(x, st, f"settle step {i+1}", e))
    xh0 = base["x"].ravel()[prep["hidden"]]
    for n, d in enumerate(model.kicks(prep, base, n_dirs, size, "ping")):
        kicked = model.compose(prep, xh0 + d)
        frames.append(panel(kicked, model.choose_states(prep, kicked).argmax(1),
                            f"ping {n//2+1}{'+' if n % 2 == 0 else '-'}: top-down kick", model.energy(prep, kicked, model.choose_states(prep, kicked))))
        s = model.settle(prep, kicked, record=True)
        for (x, st), e in zip(s["frames"], s["energies"]):
            frames.append(panel(x, st, f"ping {n//2+1}{'+' if n % 2 == 0 else '-'}: resettle", e))
    durations = [500] * len(frames)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0)
