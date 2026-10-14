"""Replace the floor of an AI-generated scene with a real scanned plank texture.

Pipeline: camera from vanishing points -> floor-plane homography -> procedural plank
layout built ONLY from real scanned plank pieces -> mip-mapped sampling -> original
scene lighting (shadows / reflections) re-applied -> feathered composite.
"""
import argparse, glob, cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--scene', default='../../assets/gemini_1772203361325.png')
ap.add_argument('--mask', default='mask4096.png')
ap.add_argument('--out', default='result.png')
ap.add_argument('--plank_w', type=float, default=0.19, help='plank width in metres')
ap.add_argument('--cam_h', type=float, default=1.2, help='camera height in metres')
ap.add_argument('--dir', choices=['vp1', 'vp2'], default='vp1', help='plank length direction')
ap.add_argument('--seed', type=int, default=7)
ap.add_argument('--light_sigma', type=float, default=14, help='shading blur (px @4096)')
ap.add_argument('--light_strength', type=float, default=1.0)
ap.add_argument('--exposure_match', type=int, default=1, help='1 = lit floor matches scan lightness')
a = ap.parse_args()
rng = np.random.default_rng(a.seed)

scene = cv2.imread(a.scene)
Hs, Ws = scene.shape[:2]
sc = Ws / 1200.0
VP1 = np.array([1605.4, 422.7]) * sc
VP2 = np.array([-727.7, 419.4]) * sc
cx, cy = Ws / 2, (VP1[1] + VP2[1]) / 2
f = np.sqrt(-((VP1[0] - cx) * (VP2[0] - cx) + (VP1[1] - cy) * (VP2[1] - cy)))
K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
Ki = np.linalg.inv(K)
def dirc(v):
    d = Ki @ np.array([v[0], v[1], 1.0]); return d / np.linalg.norm(d)
e1, e2 = dirc(VP1), dirc(VP2)
if a.dir == 'vp2': e1, e2 = e2, e1
H = K @ np.c_[e1, e2, [0, a.cam_h, 0]]          # floor (X,Y,1) -> image
Hi = np.linalg.inv(H)
print(f'f={f:.0f}px  fov={np.degrees(2*np.arctan(Ws/2/f)):.1f}deg  e1.e2={e1@e2:.4f}')

# ---------------- mask ----------------
m = cv2.imread(a.mask, 0).astype(np.float32) / 255
m = cv2.resize(m, (Ws, Hs), interpolation=cv2.INTER_LINEAR)
hard = m > 0.5
alpha = cv2.GaussianBlur(hard.astype(np.float32), (0, 0), 1.6)
ys, xs = np.nonzero(alpha > 0.002)

def to_floor(x, y):
    p = Hi @ np.vstack([x, y, np.ones_like(x)])
    return p[0] / p[2], p[1] / p[2]
xf, yf = xs.astype(np.float64), ys.astype(np.float64)
X, Y = to_floor(xf, yf)
X1, Y1 = to_floor(xf + 1, yf)
X2, Y2 = to_floor(xf, yf + 1)
fp = np.sqrt(np.hypot(X1 - X, Y1 - Y) * np.hypot(X2 - X, Y2 - Y))   # metres / pixel (geo-mean)

# ---------------- plank library (real scan pieces only) ----------------
pieces = [cv2.imread(p) for p in sorted(glob.glob('planks/p*.png'), key=lambda s: int(s.rsplit('/p', 1)[-1][:-4]))]
pw_px = np.mean([p.shape[0] for p in pieces])
m_per_px = a.plank_w / pw_px
plen = np.array([p.shape[1] * m_per_px for p in pieces])
def pyramid(img):
    lv = [img.astype(np.float32)]
    while min(lv[-1].shape[:2]) > 8: lv.append(cv2.pyrDown(lv[-1]))
    return lv
pyr = [pyramid(p) for p in pieces]
print(f'plank {a.plank_w*1000:.0f}mm x {plen.max()*1000:.0f}/{plen.min()*1000:.0f}mm, scan res {m_per_px*1000:.2f}mm/px, {len(pieces)} pieces')

# ---------------- layout: rows along X, random stagger, no repeated neighbours ----------------
r0, r1 = int(np.floor(Y.min() / a.plank_w)) - 1, int(np.floor(Y.max() / a.plank_w)) + 1
xmin, xmax = X.min() - 3, X.max() + 3
rows = {}
for r in range(r0, r1 + 1):
    x = xmin - rng.uniform(0, plen.max())
    starts, ids, flips, prev = [], [], [], -1
    while x < xmax:
        i = rng.integers(len(pieces))
        while i == prev: i = rng.integers(len(pieces))
        starts.append(x); ids.append(i); flips.append(rng.random() < .5); x += plen[i]; prev = i
    starts.append(x)
    rows[r] = (np.array(starts), np.array(ids), np.array(flips))

row = np.floor(Y / a.plank_w).astype(int)
pid = np.empty(len(X), int); flip = np.empty(len(X), bool)
u = np.empty(len(X)); dend = np.empty(len(X))
for r, (st, ids, fl) in rows.items():
    s = row == r
    if not s.any(): continue
    j = np.searchsorted(st, X[s], side='right') - 1
    pid[s] = ids[j]; flip[s] = fl[j]
    off = X[s] - st[j]
    u[s] = off / plen[ids[j]]
    dend[s] = np.minimum(off, st[j + 1] - X[s])
v = Y / a.plank_w - row
dside = np.minimum(v, 1 - v) * a.plank_w
u = np.where(flip, 1 - u, u); v = np.where(flip, 1 - v, v)

# mip level from footprint in scan pixels (slight bias to keep sharpness)
lod = np.clip(np.log2(np.maximum(fp / m_per_px, 1e-6)) - 0.3, 0, 30)
col = np.zeros((len(X), 3), np.float32)
L0 = np.floor(lod).astype(int); t = (lod - L0).astype(np.float32)
def sample(img, uu, vv):
    h, w = img.shape[:2]
    mx = (uu * w - 0.5).astype(np.float32); my = (vv * h - 0.5).astype(np.float32)
    n = len(mx); C = 4096; R = -(-n // C); pad = R * C - n
    mx = np.pad(mx, (0, pad)).reshape(R, C); my = np.pad(my, (0, pad)).reshape(R, C)
    o = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return o.reshape(-1, 3)[:n]
for i in range(len(pieces)):
    for l in np.unique(L0[pid == i]):
        s = (pid == i) & (L0 == l)
        lv = pyr[i]; a0 = lv[min(l, len(lv) - 1)]; a1 = lv[min(l + 1, len(lv) - 1)]
        c0 = sample(a0, u[s], v[s]); c1 = sample(a1, u[s], v[s])
        col[s] = c0 * (1 - t[s, None]) + c1 * t[s, None]

# micro-bevel seams between planks (anti-aliased by footprint)
seam_w = 0.0007
def seam(d):
    w = np.maximum(seam_w, fp)
    return 0.30 * (seam_w / w) * np.exp(-(d / w) ** 2)
col *= (1 - np.maximum(seam(dside), seam(dend)))[:, None].astype(np.float32)

# ---------------- lighting transfer from the original floor ----------------
lab = cv2.cvtColor(scene, cv2.COLOR_BGR2LAB).astype(np.float32)
Lin = ((lab[..., 0] / 255) ** 2.2)
hm = hard.astype(np.float32)
num = cv2.GaussianBlur(Lin * hm, (0, 0), a.light_sigma)
den = cv2.GaussianBlur(hm, (0, 0), a.light_sigma)
shade = num / np.maximum(den, 1e-4)
ref = np.median(shade[hard])
S = (shade[ys, xs] / ref) ** a.light_strength
lin = (col / 255) ** 2.2 * S[:, None]
# exposure match: the floor (excluding contact shadows) gets the same median lightness as the scan;
# shadows / window light stay relative to it. Only a neutral gain -> hue is untouched.
def Lstar(bgr):
    return cv2.cvtColor(np.clip(bgr, 0, 255).astype(np.uint8).reshape(-1, 1, 3), cv2.COLOR_BGR2LAB)[:, 0, 0].astype(np.float32)
scanL = np.median(np.concatenate([Lstar(p.reshape(-1, 3)[::7]) for p in pieces]))
lit = (S > 0.6) & (alpha[ys, xs] > 0.99)          # whole floor except contact shadows
gain = 1.0
for _ in range(4):
    curL = np.median(Lstar((lin[lit][::5] * gain) ** (1 / 2.2) * 255))
    gain *= (scanL / max(curL, 1)) ** 2.2 if a.exposure_match else 1
print(f'exposure gain {gain:.3f}')
new = np.clip((lin * gain) ** (1 / 2.2) * 255, 0, 255)

out = scene.astype(np.float32)
al = alpha[ys, xs][:, None]
out[ys, xs] = out[ys, xs] * (1 - al) + new * al
cv2.imwrite(a.out, np.clip(out, 0, 255).astype(np.uint8))
prev = cv2.resize(out.astype(np.uint8), (1200, 1200), interpolation=cv2.INTER_AREA)
cv2.imwrite(a.out.rsplit('.', 1)[0] + '_preview.jpg', prev, [cv2.IMWRITE_JPEG_QUALITY, 92])
print('done', a.out)
