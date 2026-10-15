"""Floor geometry from two independent sources, then cross-checked.

A) MoGe-2: metric point map + intrinsics  ->  RANSAC floor plane (normal, camera height)
B) Classic: LSD line segments -> vanishing points (no learning)

Output per scene (json): K, floor normal, camera height, wall direction, homography floor->image,
and agreement metrics between A and B.
"""
import sys, os, glob, json, time
import numpy as np, cv2, torch

torch.set_num_threads(8)
OUT = '/projects/sandbox/val/geo'
os.makedirs(OUT, exist_ok=True)


# ------------------------------------------------------------------ MoGe
_moge = None
def moge_infer(img_rgb):
    global _moge
    from moge.model.v2 import MoGeModel
    if _moge is None:
        p = glob.glob('/projects/sandbox/.hf/hub/models--Ruicheng--moge-2-vitl-normal/snapshots/*/model.pt')[0]
        _moge = MoGeModel.from_pretrained(p).eval()
    x = torch.tensor(img_rgb / 255.0, dtype=torch.float32).permute(2, 0, 1)
    with torch.inference_mode():
        o = _moge.infer(x, use_fp16=torch.cuda.is_available())
    return {k: (v.cpu().numpy() if torch.is_tensor(v) else v) for k, v in o.items()}


def fit_plane(P, iters=2000, thr=0.01, rng=np.random.default_rng(0)):
    """RANSAC plane n.P = c  (thr = relative to median depth)."""
    best = None
    scale = np.median(np.linalg.norm(P, axis=1))
    t = thr * scale
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a); nn = np.linalg.norm(n)
        if nn < 1e-9: continue
        n /= nn; d = n @ a
        inl = np.abs(P @ n - d) < t
        if best is None or inl.sum() > best[0].sum(): best = (inl, n, d)
    inl = best[0]
    Q = P[inl]; cen = Q.mean(0)
    _, _, Vt = np.linalg.svd(Q - cen, full_matrices=False)
    n = Vt[-1]; d = n @ cen
    if d > 0: n, d = -n, -d          # orient n towards the camera: n.P = -h, h > 0
    return n, -d, inl.mean()


# ------------------------------------------------------------------ classic VPs
def line_segments(gray):
    L = cv2.createLineSegmentDetector().detect(gray)[0]
    if L is None: return np.zeros((0, 4))
    L = L.reshape(-1, 4)
    ln = np.hypot(L[:, 2] - L[:, 0], L[:, 3] - L[:, 1])
    return L[ln > gray.shape[1] * 0.03]


def seg_dirs_cam(L, Ki):
    """Each 2D segment -> interpretation-plane normal in camera space."""
    p = np.c_[L[:, :2], np.ones(len(L))] @ Ki.T
    q = np.c_[L[:, 2:], np.ones(len(L))] @ Ki.T
    m = np.cross(p, q); return m / np.linalg.norm(m, axis=1)[:, None]


def wall_direction_from_lines(L, K, up):
    """Given floor normal (camera space), find the in-plane horizontal direction best supported by
    image lines (Manhattan: d and up x d). Returns unit vector, support, angular residual."""
    Ki = np.linalg.inv(K)
    m = seg_dirs_cam(L, Ki)
    w = np.hypot(L[:, 2] - L[:, 0], L[:, 3] - L[:, 1])
    # basis of the floor plane
    a = np.cross(up, [1, 0, 0]); a /= np.linalg.norm(a); b = np.cross(up, a)
    best = None
    for th in np.radians(np.arange(0, 90, 0.25)):
        d1 = np.cos(th) * a + np.sin(th) * b; d2 = np.cross(up, d1)
        # a 3D direction d is consistent with a segment if d is ~in its interpretation plane (m.d = 0)
        e = np.minimum(np.abs(m @ d1), np.abs(m @ d2))
        s = (w * (e < np.sin(np.radians(0.7)))).sum()
        if best is None or s > best[1]: best = (d1, s, th)
    d1 = best[0]; d2 = np.cross(up, d1)
    e = np.minimum(np.abs(m @ d1), np.abs(m @ d2)); inl = e < np.sin(np.radians(0.7))
    vert = np.abs(m @ up) < np.sin(np.radians(0.7))
    return d1, float(w[inl].sum() / w.sum()), float(w[vert].sum() / w.sum()), inl, vert


def classic_vps(L, W, H):
    """Purely classic: RANSAC 2 dominant horizontal-ish VPs (as in v1) -> horizon, focal."""
    x1, y1, x2, y2 = L.T
    ang = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
    Hs = L[np.abs(ang - 90) > 10]
    w = np.hypot(Hs[:, 2] - Hs[:, 0], Hs[:, 3] - Hs[:, 1])
    lines = np.cross(np.c_[Hs[:, :2], np.ones(len(Hs))], np.c_[Hs[:, 2:], np.ones(len(Hs))])
    lines /= np.linalg.norm(lines[:, :2], axis=1)[:, None]
    rng = np.random.default_rng(1); rem = np.ones(len(Hs), bool); vps = []
    mid = (Hs[:, :2] + Hs[:, 2:]) / 2; dvec = Hs[:, 2:] - Hs[:, :2]; dvec /= np.linalg.norm(dvec, axis=1)[:, None]
    for _ in range(2):
        idx = np.where(rem)[0]
        if len(idx) < 4: break
        best = None
        for _ in range(3000):
            i, j = rng.choice(idx, 2, replace=False)
            v = np.cross(lines[i], lines[j])
            if abs(v[2]) < 1e-12: continue
            v = v / v[2]
            t = v[:2] - mid[idx]; t /= np.linalg.norm(t, axis=1)[:, None] + 1e-9
            err = np.degrees(np.arccos(np.clip(np.abs((dvec[idx] * t).sum(1)), 0, 1)))
            inl = idx[err < 1.0]; sc = w[inl].sum()
            if best is None or sc > best[0]: best = (sc, inl)
        inl = best[1]
        A = lines[inl] * w[inl][:, None]
        v = np.linalg.svd(A)[2][-1]; v = v / v[2]
        vps.append(v[:2]); rem[inl] = False
    return vps


# ------------------------------------------------------------------ hybrid refinement
def rot_up(pitch, roll):
    """camera-space 'up' (towards camera side of floor) for given pitch/roll (deg). Image y is down."""
    p, r = np.radians(pitch), np.radians(roll)
    u = np.array([0.0, -1.0, 0.0])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return Rz @ Rx @ u


def up_to_pr(u):
    # inverse of rot_up (approx, small roll)
    pitch = np.degrees(np.arcsin(np.clip(-u[2], -1, 1)))
    roll = np.degrees(np.arctan2(u[0], -u[1]))
    return pitch, roll


def manhattan_cost(params, L, w, W, H, cxcy):
    f, pitch, roll, yaw = params
    K = np.array([[f, 0, cxcy[0]], [0, f, cxcy[1]], [0, 0, 1]])
    m = seg_dirs_cam(L, np.linalg.inv(K))
    up = rot_up(pitch, roll)
    a = np.cross(up, [1, 0, 0]); a /= np.linalg.norm(a); b = np.cross(up, a)
    t = np.radians(yaw); d1 = np.cos(t) * a + np.sin(t) * b; d2 = np.cross(up, d1)
    e = np.min(np.abs(np.c_[m @ d1, m @ d2, m @ up]), axis=1)
    s = np.sin(np.radians(1.0))
    return float((w * (e ** 2 / (e ** 2 + s ** 2))).sum() / w.sum())   # robust (Cauchy-like)


def refine(L, K0, up0, W, H):
    from scipy.optimize import minimize
    w = np.hypot(L[:, 2] - L[:, 0], L[:, 3] - L[:, 1])
    cxcy = (K0[0, 2], K0[1, 2])
    p0, r0 = up_to_pr(up0)
    # sanity: rot_up(p0,r0) should match up0
    best = None
    for yaw in np.arange(0, 90, 1.0):
        c = manhattan_cost([K0[0, 0], p0, r0, yaw], L, w, W, H, cxcy)
        if best is None or c < best[0]: best = (c, yaw)
    x0 = np.array([K0[0, 0], p0, r0, best[1]])
    # priors keep us near MoGe when lines are scarce
    def obj(x):
        prior = ((x[0] / K0[0, 0] - 1) / 0.08) ** 2 + ((x[1] - p0) / 2.0) ** 2 + ((x[2] - r0) / 1.5) ** 2
        return manhattan_cost(x, L, w, W, H, cxcy) + 0.002 * prior
    r = minimize(obj, x0, method='Nelder-Mead', options=dict(xatol=1e-3, fatol=1e-6, maxiter=3000))
    return r.x, manhattan_cost(x0, L, w, W, H, cxcy), manhattan_cost(r.x, L, w, W, H, cxcy), (p0, r0)


# ------------------------------------------------------------------ shift-lens camera model
def cost_shift(params, L, w, cx):
    """params: f, cy, pitch, roll, yaw.  Principal point may be shifted vertically (architectural
    'rise'), so a high horizon no longer forces a tilted camera."""
    f, cy, pitch, roll, yaw = params
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    return manhattan_cost([f, pitch, roll, yaw], L, w, None, None, (cx, cy)) if False else \
        _mc(K, pitch, roll, yaw, L, w)


def _mc(K, pitch, roll, yaw, L, w):
    m = seg_dirs_cam(L, np.linalg.inv(K))
    up = rot_up(pitch, roll)
    a = np.cross(up, [1, 0, 0]); a /= np.linalg.norm(a); b = np.cross(up, a)
    t = np.radians(yaw); d1 = np.cos(t) * a + np.sin(t) * b; d2 = np.cross(up, d1)
    e = np.min(np.abs(np.c_[m @ d1, m @ d2, m @ up]), axis=1)
    s = np.sin(np.radians(1.0))
    return float((w * (e ** 2 / (e ** 2 + s ** 2))).sum() / w.sum())


def refine_shift(L, K0, up0, W, H, hor0):
    """Search both hypotheses (tilted camera vs. shifted principal point) and keep the better one."""
    from scipy.optimize import minimize
    w = np.hypot(L[:, 2] - L[:, 0], L[:, 3] - L[:, 1])
    cx = W / 2
    p0, r0 = up_to_pr(up0)
    f0 = K0[0, 0]
    inits = []
    for pitch_i, cy_i in ((p0, H / 2), (0.0, hor0)):            # tilt-hypothesis, shift-hypothesis
        best = None
        for yaw in np.arange(0, 90, 1.0):
            c = _mc(np.array([[f0, 0, cx], [0, f0, cy_i], [0, 0, 1]]), pitch_i, r0, yaw, L, w)
            if best is None or c < best[0]: best = (c, yaw)
        inits.append(np.array([f0, cy_i, pitch_i, r0, best[1]]))
    def obj(x):
        f, cy, pitch, roll, yaw = x
        prior = ((f / f0 - 1) / 0.15) ** 2 + ((cy - H / 2) / (0.5 * H)) ** 2
        return _mc(np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]]), pitch, roll, yaw, L, w) + 0.002 * prior
    sols = [minimize(obj, x0, method='Nelder-Mead', options=dict(xatol=1e-3, fatol=1e-7, maxiter=4000)) for x0 in inits]
    s = min(sols, key=lambda r: r.fun)
    return s.x, [r.fun for r in sols]


def height_from_moge(P, mask, K, up):
    """re-derive camera height for the refined camera from MoGe's depth along each floor pixel ray."""
    ys, xs = np.nonzero(mask)
    z = P[ys, xs, 2]; ok = np.isfinite(z) & (z > 0)
    ys, xs, z = ys[ok], xs[ok], z[ok]
    rays = np.c_[xs + 0.5, ys + 0.5, np.ones(len(xs))] @ np.linalg.inv(K).T     # z = 1
    hh = -z * (rays @ up)
    return float(np.median(hh))


# ------------------------------------------------------------------ main
def run(f, floor_mask_path):
    name = os.path.basename(f)[:-4]
    bgr = cv2.imread(f); H, W = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    fm = cv2.imread(floor_mask_path, 0) > 127
    t = time.time(); o = moge_infer(rgb); tm = time.time() - t
    P = o['points']; Kn = o['intrinsics']             # normalized intrinsics
    K = Kn.copy(); K[0] *= W; K[1] *= H
    valid = o['mask'].astype(bool) & fm & np.isfinite(P).all(-1)
    # erode floor mask to avoid edges
    valid &= cv2.erode(fm.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    pts = P[valid][::max(1, valid.sum() // 60000)]
    n, h, inl_ratio = fit_plane(pts)
    up = n                                             # camera-space floor normal pointing to camera side
    # horizon line in image: l = K^-T n
    hl = np.linalg.inv(K).T @ n
    hor_y_center = -(hl[0] * W / 2 + hl[2]) / hl[1]
    tilt = np.degrees(np.arcsin(np.clip(n[2], -1, 1)))  # camera pitch relative to floor
    roll = np.degrees(np.arctan2(-hl[0], hl[1]))

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    L = line_segments(gray)
    d1, manh_support, vert_support, inl, vert = wall_direction_from_lines(L, K, up)

    # hybrid: refine MoGe camera with Manhattan line evidence
    x, c0, c1, (p0, r0) = refine(L, K, up, W, H)
    fR, pR, rR, yR = x
    upR = rot_up(pR, rR)
    KR = np.array([[fR, 0, K[0, 2]], [0, fR, K[1, 2]], [0, 0, 1]])
    hlR = np.linalg.inv(KR).T @ upR
    cls = dict(f_px=float(fR), pitch=float(pR), roll=float(rR), yaw=float(yR),
               moge_pitch=float(p0), moge_roll=float(r0),
               cost_moge=c0, cost_refined=c1,
               horizon_y_center=float(-(hlR[0] * W / 2 + hlR[2]) / hlR[1]),
               up=upR.round(5).tolist())
    # shift-lens model
    xs_, costs = refine_shift(L, K, up, W, H, cls['horizon_y_center'])
    fS, cyS, pS, rS, yS = xs_
    KS = np.array([[fS, 0, W / 2], [0, fS, cyS], [0, 0, 1]])
    upS = rot_up(pS, rS)
    hlS = np.linalg.inv(KS).T @ upS
    hS = height_from_moge(P, valid, KS, upS)
    cls['shift'] = dict(f_px=float(fS), cy=float(cyS), pitch=float(pS), roll=float(rS), yaw=float(yS),
                        cost_tilt_hyp=costs[0], cost_shift_hyp=costs[1], cam_height_m=hS,
                        horizon_y_center=float(-(hlS[0] * W / 2 + hlS[2]) / hlS[1]), up=upS.round(6).tolist())
    np.save(f'{OUT}/{name}_KS.npy', KS)
    # MoGe-predicted VPs for the wall directions (to compare with classic ones)
    def vp(d):
        p = K @ d; return (p[:2] / p[2]).round(1).tolist() if abs(p[2]) > 1e-9 else None
    d2 = np.cross(up, d1)
    res = dict(name=name, W=W, H=H, moge_s=round(tm, 1),
               f_px=float(K[0, 0]), fov_x=float(np.degrees(2 * np.arctan(W / 2 / K[0, 0]))),
               cx=float(K[0, 2]), cy=float(K[1, 2]),
               normal=n.round(4).tolist(), cam_height_m=float(h), plane_inlier=float(inl_ratio),
               pitch_deg=float(tilt), roll_deg=float(roll), horizon_y_center=float(hor_y_center),
               wall_dir=d1.round(4).tolist(), wall_vps=[vp(d1), vp(d2)],
               line_support_manhattan=manh_support, line_support_vertical=vert_support,
               classic=cls)
    json.dump(res, open(f'{OUT}/{name}.json', 'w'), indent=1)
    np.save(f'{OUT}/{name}_K.npy', K)
    # visual: lines coloured by Manhattan class + horizon
    vis = bgr.copy()
    for s, a_, b_ in zip(L, inl, vert):
        c = (0, 200, 0) if b_ else ((0, 0, 255) if a_ else (160, 160, 160))
        cv2.line(vis, tuple(s[:2].astype(int)), tuple(s[2:].astype(int)), c, 2)
    x0, x1 = 0, W
    y0 = -(hl[0] * x0 + hl[2]) / hl[1]; y1 = -(hl[0] * x1 + hl[2]) / hl[1]
    cv2.line(vis, (x0, int(y0)), (x1, int(y1)), (255, 0, 255), 2)
    cv2.imwrite(f'{OUT}/{name}_lines.jpg', cv2.resize(vis, (W // 2, H // 2)))
    return res


if __name__ == '__main__':
    files = sys.argv[1:] or sorted(glob.glob('/projects/sandbox/val/scenes/*.png'))
    for f in files:
        name = os.path.basename(f)[:-4]
        r = run(f, f'/projects/sandbox/val/sam3/{name}_floor.png')
        c = r['classic']
        print(f"{name}: moge {r['moge_s']}s | MoGe f {r['f_px']:.0f} pitch {c['moge_pitch']:.2f} roll {c['moge_roll']:.2f} horizon {r['horizon_y_center']:.0f} h {r['cam_height_m']:.2f}m"
              f" || refined f {c['f_px']:.0f} ({(c['f_px']/r['f_px']-1)*100:+.1f}%) pitch {c['pitch']:.2f} roll {c['roll']:.2f} horizon {c['horizon_y_center']:.0f}"
              f" | line cost {c['cost_moge']:.3f}->{c['cost_refined']:.3f}", flush=True)
        S = c['shift']
        print(f"      SHIFT model: f {S['f_px']:.0f} cy {S['cy']:.0f} pitch {S['pitch']:+.2f} roll {S['roll']:+.2f} horizon {S['horizon_y_center']:.0f} h {S['cam_height_m']:.2f}m | cost tilt-hyp {S['cost_tilt_hyp']:.3f} shift-hyp {S['cost_shift_hyp']:.3f}", flush=True)
