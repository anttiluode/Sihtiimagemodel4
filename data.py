"""Three data sources, all 32x32 grayscale in [0, 1].

1. synthetic : the SihtiImageModel3 contour world (same generator, seeds and
               damage as Gate R1, so numbers are directly comparable).
2. natural   : crops of scikit-image's bundled photographs. Training/validation
               crops come from one set of photographs (top 75% / bottom 25% of
               each), test crops from DIFFERENT photographs. No download needed.
3. folders   : your own disjoint train/unseen folders (e.g. THINGS images).
"""
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


# ------------------------------------------------------------ synthetic (R1)
def contour_image(seed):
    """Copied unchanged from anttiluode/SihtiImageModel3 data.py."""
    rng = np.random.default_rng(seed)
    im = Image.new("L", (32, 32), 0)
    d = ImageDraw.Draw(im)
    family = int(rng.integers(0, 5))
    thick = int(rng.integers(2, 5))
    x = int(rng.integers(6, 11))
    y = int(rng.integers(6, 11))
    w = int(rng.integers(12, 19))
    h = int(rng.integers(12, 19))
    if family == 0:
        d.arc((x, y, x+w, y+h), start=int(rng.integers(-20, 20)), end=int(rng.integers(300, 360)), fill=255, width=thick)
    elif family == 1:
        d.rectangle((x, y, x+w, y+h), outline=255, width=thick)
    elif family == 2:
        d.line([(x, y+h), (x+w//2, y), (x+w, y+h)], fill=255, width=thick, joint="curve")
    elif family == 3:
        d.line([(x, y), (x+w, y+h//2), (x, y+h)], fill=255, width=thick, joint="curve")
    else:
        d.line([(x, y+h), (x+w//3, y+1), (x+2*w//3, y+h-1), (x+w, y)], fill=255, width=thick, joint="curve")
    if rng.random() < .5:
        d.line((x+2, y+h//2, x+w-2, y+h//2), fill=255, width=thick)
    return np.asarray(im.filter(ImageFilter.GaussianBlur(.45)), dtype=np.float32)/255.


def synthetic(n, seed):
    return np.array([contour_image(seed*100000 + i) for i in range(n)])


def damage(img, seed=0, kind="gap"):
    """Same masks as Gate R1: a 5-px vertical gap or an 8x8 hole."""
    rng = np.random.default_rng(seed)
    mask = np.ones_like(img, bool)
    if kind == "gap":
        start = int(rng.integers(11, 17))
        mask[:, start:start+5] = False
    elif kind == "hole":
        x = int(rng.integers(11, 18))
        y = int(rng.integers(11, 18))
        mask[y:y+8, x:x+8] = False
    else:
        raise ValueError("unknown damage type")
    obs = np.where(mask, img, .5)
    return obs, mask


def damaged_set(images, seed):
    """Alternate gap/hole exactly as R1 did (seed*10 + i)."""
    out = []
    for i, t in enumerate(images):
        kind = "gap" if i % 2 == 0 else "hole"
        obs, mask = damage(t, seed=seed*10 + i, kind=kind)
        out.append((t, obs, mask, kind))
    return out


# ------------------------------------------------------------ natural crops
TRAIN_PHOTOS = ["astronaut", "chelsea", "coins", "moon", "rocket", "immunohistochemistry",
                "hubble_deep_field", "grass", "gravel", "brick", "retina", "page", "text",
                "colorwheel", "microaneurysms"]
TEST_PHOTOS = ["camera", "coffee", "clock"]


def _photo(name):
    import skimage.data
    from skimage.color import rgb2gray, rgba2rgb
    im = getattr(skimage.data, name)()
    if im.ndim == 3 and im.shape[2] == 4:
        im = rgba2rgb(im)
    if im.ndim == 3:
        im = rgb2gray(im)
    im = np.asarray(im, float)
    if im.max() > 1.5:
        im = im / 255.
    return im


def _crops(img, n, rng, rows=None, min_std=.05, tries=4000):
    """Random 32x32 crops at three downscalings; rows=(lo, hi) restricts the
    crop to a horizontal band (fractions of height) so train/val never overlap."""
    from skimage.transform import rescale
    scaled = {s: rescale(img, s, anti_aliasing=True) for s in (.25, .35, .5)}
    out = []
    for _ in range(tries):
        if len(out) >= n:
            break
        small = scaled[float(rng.choice([.25, .35, .5]))]
        h, w = small.shape
        lo, hi = (0, h) if rows is None else (int(rows[0]*h), int(rows[1]*h))
        if hi - lo < 32 or w < 32:
            continue
        y = int(rng.integers(lo, hi - 31))
        x = int(rng.integers(0, w - 31))
        c = small[y:y+32, x:x+32]
        if c.std() >= min_std:
            out.append(c.astype(np.float32))
    return out


def natural(n_train=1500, n_val=64, n_test=64, seed=0):
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for name in TRAIN_PHOTOS:
        img = _photo(name)
        train += _crops(img, n_train // len(TRAIN_PHOTOS) + 1, rng, rows=(0., .75))
        val += _crops(img, n_val // len(TRAIN_PHOTOS) + 1, rng, rows=(.75, 1.))
    for name in TEST_PHOTOS:
        test += _crops(_photo(name), n_test // len(TEST_PHOTOS) + 1, rng)
    pick = lambda a, n: np.array(a)[rng.permutation(len(a))[:n]]
    return pick(train, n_train), pick(val, n_val), pick(test, n_test)


# ------------------------------------------------------------ your folders
def load_images(folder, max_images=None):
    """PNG/JPG files -> centre crop -> 32x32 luminance. No labels."""
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Not a folder: {root}")
    paths = sorted(x for x in root.rglob("*")
                   if x.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp"))
    if max_images:
        paths = paths[:max_images]
    if not paths:
        raise ValueError(f"No images found under {root}")
    images = []
    for path in paths:
        with Image.open(path) as original:
            img = original.convert("L")
            w, h = img.size
            side = min(w, h)
            img = img.crop(((w-side)//2, (h-side)//2, (w-side)//2 + side, (h-side)//2 + side))
            img = img.resize((32, 32), Image.Resampling.LANCZOS)
            images.append(np.asarray(img, dtype=np.float32)/255.)
    return np.asarray(images)


def check_disjoint(train, test):
    seen = {hashlib.sha256(np.ascontiguousarray(x).tobytes()).digest() for x in train}
    if any(hashlib.sha256(np.ascontiguousarray(x).tobytes()).digest() in seen for x in test):
        raise ValueError("identical image in train and evaluation sets")
