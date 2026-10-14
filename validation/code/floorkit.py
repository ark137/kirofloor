"""floorkit: lay REAL plank photos onto a floor plane and composite into a scene.

Geometry : 3x3 homography floor-metres -> image pixels (from the camera model).
Layouts  : 'straight' (工字拼, random stagger) and 'herringbone' (人字拼).
Textures : only real plank images, 180-degree rotation allowed, never mirrored/stretched.
Lighting : original floor luminance (blurred, masked) multiplied in linear light,
           exposure matched so the lit floor has the scan's median lightness.
"""
import glob
import numpy as np, cv2


# ----------------------------------------------------------------------------- library
class PlankLib:
    def __init__(self, folder, plank_w, plank_len=None):
        files = sorted(glob.glob(f'{folder}/p*.png'), key=lambda s: int(s.rsplit('/p', 1)[1][:-4]))
        self.img = [cv2.imread(f) for f in files]
        self.x_scale = 1.0
        if plank_len:   # correct anisotropic scans: longest piece must have the real length/width ratio
            r_px = max(im.shape[1] / im.shape[0] for im in self.img)
            self.x_scale = (plank_len / plank_w) / r_px
            if abs(self.x_scale - 1) > 0.02:
                self.img = [cv2.resize(im, (round(im.shape[1] * self.x_scale), im.shape[0]),
                                       interpolation=cv2.INTER_AREA) for im in self.img]
        self.plank_w = plank_w
        self.mpp = [plank_w / im.shape[0] for im in self.img]                  # metres / px per piece
        self.len = np.array([im.shape[1] * m for im, m in zip(self.img, self.mpp)])
        self.pyr = []
        for im in self.img:
            lv = [im.astype(np.float32)]
            while min(lv[-1].shape[:2]) > 8: lv.append(cv2.pyrDown(lv[-1]))
            self.pyr.append(lv)
        Ls = [cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 0][::5, ::5].ravel() for im in self.img]
        self.median_L = float(np.median(np.concatenate(Ls)))

    def __len__(self): return len(self.img)


def hash01(*keys):
    h = np.uint64(1469598103934665603)
    for k in keys:
        h = (h ^ np.asarray(k).astype(np.int64).astype(np.uint64)) * np.uint64(1099511628211)
        h ^= h >> np.uint64(29)
    return (h % np.uint64(1 << 30)).astype(np.float64) / float(1 << 30)


# ----------------------------------------------------------------------------- layouts
def layout_straight(X, Y, lib, seed=0, min_stagger=0.25):
    """Rows along X. Each row: random start offset, consecutive pieces (no identical neighbours).
    Returns pid, u, v, d_side, d_end (metres)."""
    rng = np.random.default_rng(seed)
    w = lib.plank_w
    row = np.floor(Y / w).astype(np.int64)
    r0, r1 = row.min(), row.max()
    xmin, xmax = X.min() - 0.1, X.max() + 0.1
    pid = np.zeros(len(X), np.int64); u = np.zeros(len(X)); dend = np.zeros(len(X)); rot = np.zeros(len(X), bool)
    prev_joints = None
    for r in range(r0, r1 + 1):
        s = row == r
        if not s.any(): continue
        # choose a start offset whose joints are not too close to the previous row's joints
        for _ in range(30):
            x = xmin - rng.uniform(0, lib.len.max()); starts, ids, last = [], [], -1
            while x < xmax:
                i = int(rng.integers(len(lib)))
                while i == last and len(lib) > 1: i = int(rng.integers(len(lib)))
                starts.append(x); ids.append(i); x += lib.len[i]; last = i
            starts.append(x)
            j = np.array(starts)
            if prev_joints is None: break
            dmin = np.min(np.abs(j[:, None] - prev_joints[None, :]), axis=1)
            if (dmin[(j > xmin) & (j < xmax)] > min_stagger * lib.len.mean() * 0.5).all(): break
        prev_joints = j
        st = j; ids = np.array(ids); fl = rng.random(len(ids)) < .5
        k = np.searchsorted(st, X[s], side='right') - 1
        pid[s] = ids[k]; rot[s] = fl[k]
        off = X[s] - st[k]
        u[s] = off / lib.len[ids[k]]
        dend[s] = np.minimum(off, st[k + 1] - X[s])
    v = Y / w - row
    dside = np.minimum(v, 1 - v) * w
    u = np.where(rot, 1 - u, u); v = np.where(rot, 1 - v, v)
    return pid, u, v, dside, dend


def layout_herringbone(X, Y, lib, seed=0):
    """Classic herringbone, zig-zag axis along +X. Plank L x W from the library (median length)."""
    W = lib.plank_w; L = float(np.median(lib.len))
    # work in a frame rotated 45deg so the axis-aligned construction's diagonal maps to +X
    c = np.sqrt(.5)
    px = c * (X - Y); py = c * (X + Y)          # inverse rotation of the (1,1) axis onto X
    P = np.c_[px, py]
    T = np.array([[W, L], [W, -L]])             # columns: t1=(W,W), t2=(L,-L)
    Ti = np.linalg.inv(T)
    base = np.floor(P @ Ti.T)
    pid = np.full(len(X), -1, np.int64); u = np.zeros(len(X)); v = np.zeros(len(X))
    dside = np.zeros(len(X)); dend = np.zeros(len(X))
    ni = int(np.ceil(L / W)) + 2
    for di in range(-ni, ni + 1):
        for dj in range(-2, 3):
            ij = base + [di, dj]
            o = ij @ T.T
            q = P - o
            # horizontal plank [0,L]x[0,W]
            inH = (pid < 0) & (q[:, 0] >= 0) & (q[:, 0] < L) & (q[:, 1] >= 0) & (q[:, 1] < W)
            # vertical plank [0,W]x[W,W+L]
            inV = (pid < 0) & (q[:, 0] >= 0) & (q[:, 0] < W) & (q[:, 1] >= W) & (q[:, 1] < W + L)
            for m_, kind in ((inH, 0), (inV, 1)):
                if not m_.any(): continue
                hsh = hash01(ij[m_, 0] + 7919 * seed, ij[m_, 1], kind)
                k = np.minimum((hsh * len(lib)).astype(np.int64), len(lib) - 1)
                rot = hash01(ij[m_, 0], ij[m_, 1] + 104729, kind + 3) < .5
                if kind == 0:
                    uu = q[m_, 0] / L; vv = q[m_, 1] / W
                else:
                    uu = (q[m_, 1] - W) / L; vv = 1 - q[m_, 0] / W
                ds = np.minimum(vv, 1 - vv) * W; de = np.minimum(uu, 1 - uu) * L
                uu = np.where(rot, 1 - uu, uu); vv = np.where(rot, 1 - vv, vv)
                pid[m_] = k; u[m_] = uu; v[m_] = vv; dside[m_] = ds; dend[m_] = de
    miss = pid < 0
    if miss.any(): print('herringbone: uncovered px', int(miss.sum()))
    pid[miss] = 0
    # pieces may differ slightly in length from the median: sample only the first L metres
    # of each piece (centre-cropped) so nothing is stretched
    scale = np.array([L / l if l > L else 1.0 for l in lib.len])
    off = (1 - scale) / 2
    u = off[pid] + u * scale[pid]
    return pid, u, v, dside, dend


# ----------------------------------------------------------------------------- render
def _sample(img, uu, vv):
    h, w = img.shape[:2]
    mx = (uu * w - 0.5).astype(np.float32); my = (vv * h - 0.5).astype(np.float32)
    n = len(mx); C = 4096; R = -(-n // C); pad = R * C - n
    mx = np.pad(mx, (0, pad)).reshape(R, C); my = np.pad(my, (0, pad)).reshape(R, C)
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).reshape(-1, 3)[:n]


def guided_alpha(scene, mask, r=4, eps=2e-3):
    """Snap a slightly-off binary mask to real image edges (He et al. guided filter, colour guide)."""
    I = scene.astype(np.float32) / 255; p = mask.astype(np.float32)
    box = lambda x: cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1))
    mI = box(I); mp = box(p)
    cov = box(I * p[..., None]) - mI * mp[..., None]
    var = np.zeros(I.shape[:2] + (3, 3), np.float32)
    for i in range(3):
        for j in range(3):
            var[..., i, j] = box(I[..., i] * I[..., j]) - mI[..., i] * mI[..., j]
    var += eps * np.eye(3, dtype=np.float32)
    a = np.linalg.solve(var, cov[..., None])[..., 0]
    b = mp - (a * mI).sum(-1)
    q = (box(a) * I).sum(-1) + box(b)
    # only allow changes in a thin band around the original edge
    band = cv2.dilate(p, np.ones((2 * r + 1,) * 2)) - cv2.erode(p, np.ones((2 * r + 1,) * 2))
    return np.clip(np.where(band > 0, q, p), 0, 1)


def render(scene, floor_mask, Hm, lib, layout='straight', seed=0, light_sigma_frac=0.001,
           light_strength=1.0, light_color=0.0, seam=0.25, seam_w=0.0008, rotate_deg=0.0, refine_edges=True,
           detail_thresh=0.08, coarse_frac=0.006, highlight_knee=0.7):
    """scene: BGR uint8; floor_mask: bool HxW; Hm: floor(m)->image homography."""
    Hs, Ws = floor_mask.shape
    if refine_edges:
        alpha = guided_alpha(scene, floor_mask, r=max(2, Ws // 512))
        alpha = np.clip((alpha - 0.15) / 0.7, 0, 1)
    else:
        alpha = cv2.GaussianBlur(floor_mask.astype(np.float32), (0, 0), 0.8)
    ys, xs = np.nonzero(alpha > 0.002)
    Hi = np.linalg.inv(Hm)
    def to_floor(x, y):
        p = Hi @ np.vstack([x, y, np.ones_like(x)]); return p[0] / p[2], p[1] / p[2]
    xf, yf = xs + 0.5, ys + 0.5
    X, Y = to_floor(xf, yf); X1, Y1 = to_floor(xf + 1, yf); X2, Y2 = to_floor(xf, yf + 1)
    fp = np.sqrt(np.hypot(X1 - X, Y1 - Y) * np.hypot(X2 - X, Y2 - Y))
    if rotate_deg:
        t = np.radians(rotate_deg); X, Y = np.cos(t) * X - np.sin(t) * Y, np.sin(t) * X + np.cos(t) * Y
    if layout == 'herringbone':
        pid, u, v, dside, dend = layout_herringbone(X, Y, lib, seed)
    else:
        pid, u, v, dside, dend = layout_straight(X, Y, lib, seed)
    mpp = np.array(lib.mpp)[pid]
    lod = np.clip(np.log2(np.maximum(fp / mpp, 1e-6)) - 0.3, 0, 30)
    L0 = np.floor(lod).astype(int); tt = (lod - L0).astype(np.float32)
    col = np.zeros((len(X), 3), np.float32)
    for i in range(len(lib)):
        for l in np.unique(L0[pid == i]):
            s = (pid == i) & (L0 == l)
            lv = lib.pyr[i]
            c0 = _sample(lv[min(l, len(lv) - 1)], u[s], v[s]); c1 = _sample(lv[min(l + 1, len(lv) - 1)], u[s], v[s])
            col[s] = c0 * (1 - tt[s, None]) + c1 * tt[s, None]
    # micro-bevel seams, anti-aliased by footprint
    def sm(d):
        w = np.maximum(seam_w, fp); return seam * (seam_w / w) * np.exp(-(d / w) ** 2)
    col *= (1 - np.maximum(sm(dside), sm(dend)))[:, None].astype(np.float32)

    # ---- lighting from the original floor
    sig = light_sigma_frac * Ws
    lin_scene = (scene.astype(np.float32) / 255) ** 2.2
    lum = lin_scene @ np.array([0.0722, 0.7152, 0.2126], np.float32)        # BGR -> Y
    hm = floor_mask.astype(np.float32)
    def mblur(x, s):
        return cv2.GaussianBlur(x * hm, (0, 0), s) / np.maximum(cv2.GaussianBlur(hm, (0, 0), s), 1e-4)
    den = np.maximum(cv2.GaussianBlur(hm, (0, 0), sig), 1e-4)
    if detail_thresh > 0:
        # two-scale: coarse light + only the STRONG fine detail (shadow/sun edges);
        # weak fine detail = the old floor's own texture (concrete mottling) -> suppressed
        coarse = mblur(lum, coarse_frac * Ws)
        fine = mblur(lum, sig)
        d = np.log(np.maximum(fine, 1e-5) / np.maximum(coarse, 1e-5))
        d = d * (d * d / (d * d + detail_thresh ** 2))
        shade = coarse * np.exp(d)
    else:
        shade = mblur(lum, sig)
    ref = np.median(shade[floor_mask])
    S = (shade[ys, xs] / ref) ** light_strength
    lin = (col / 255) ** 2.2 * S[:, None]
    if light_color > 0:          # optional: carry the light's tint (warm sun / cool shade)
        chrom = np.stack([cv2.GaussianBlur(lin_scene[..., c] * hm, (0, 0), sig * 4) / den for c in range(3)], -1)
        chrom = chrom[ys, xs]; chrom /= (chrom @ np.array([0.0722, 0.7152, 0.2126]))[:, None] + 1e-6
        med = np.median(chrom[floor_mask[ys, xs]], axis=0)
        lin *= (chrom / med) ** light_color
    # exposure match on the lit floor
    lit = (S > 0.6) & (alpha[ys, xs] > 0.99)
    def Lstar(lin_bgr):
        b = np.clip(lin_bgr ** (1 / 2.2) * 255, 0, 255).astype(np.uint8).reshape(-1, 1, 3)
        return cv2.cvtColor(b, cv2.COLOR_BGR2LAB)[:, 0, 0].astype(np.float32)
    gain = 1.0
    for _ in range(4):
        gain *= (lib.median_L / max(np.median(Lstar(lin[lit][::3] * gain)), 1)) ** 2.2
    lin = lin * gain
    if highlight_knee < 1:     # soft shoulder on luminance: keep grain in sun patches, hue unchanged
        Yl = lin @ np.array([0.0722, 0.7152, 0.2126], np.float32)
        k = highlight_knee
        Yc = np.where(Yl > k, k + (1 - k) * (1 - np.exp(-(Yl - k) / (1 - k))), Yl)
        lin = lin * (Yc / np.maximum(Yl, 1e-6))[:, None]
    new = np.clip(lin ** (1 / 2.2) * 255, 0, 255)
    out = scene.astype(np.float32)
    a = alpha[ys, xs][:, None]
    out[ys, xs] = out[ys, xs] * (1 - a) + new * a
    return np.clip(out, 0, 255).astype(np.uint8)
