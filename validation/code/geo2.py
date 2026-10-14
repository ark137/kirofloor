"""Geometry v2: shift-lens camera, MoGe-anchored focal, held-out line validation.

Camera: square pixels, principal point x = W/2, free principal point y (architectural rise),
pitch/roll/yaw. MoGe gives focal + floor plane + metric scale; lines refine.
Validation: fit with lines ABOVE the horizon (+ verticals), test angular error on horizontal
segments BELOW the horizon near the floor (furniture bottoms, wall bases, rugs) -> held out.
"""
import sys, os, glob, json
import numpy as np, cv2
from scipy.optimize import minimize

src = open('/projects/sandbox/val/geo.py').read().split('# ------------------------------------------------------------------ main')[0]
exec(src)  # reuse: moge_infer, fit_plane, line_segments, seg_dirs_cam, rot_up, up_to_pr, _mc

D = '/projects/sandbox/val'
os.makedirs(f'{D}/geo2', exist_ok=True)


def cam_K(f, cy, W):
    return np.array([[f, 0, W / 2], [0, f, cy], [0, 0, 1.0]])


def dirs(up, yaw):
    a = np.cross(up, [1, 0, 0]); a /= np.linalg.norm(a); b = np.cross(up, a)
    t = np.radians(yaw); d1 = np.cos(t) * a + np.sin(t) * b
    return d1, np.cross(up, d1)


def fit(L, W, H, f0, p0, r0, cy0, f_sigma=0.06):
    w = np.hypot(L[:, 2] - L[:, 0], L[:, 3] - L[:, 1])
    best = None
    for pitch_i, cy_i in ((p0, H / 2), (0.0, cy0)):
        for yaw in np.arange(0, 90, 1.0):
            c = _mc(cam_K(f0, cy_i, W), pitch_i, r0, yaw, L, w)
            if best is None or c < best[0]: best = (c, np.array([f0, cy_i, pitch_i, r0, yaw]))
    def obj(x):
        f, cy, pitch, roll, yaw = x
        prior = ((f / f0 - 1) / f_sigma) ** 2 + ((cy - H / 2) / (0.5 * H)) ** 2 + (roll / 3) ** 2
        return _mc(cam_K(f, cy, W), pitch, roll, yaw, L, w) + 0.004 * prior
    r = minimize(obj, best[1], method='Nelder-Mead', options=dict(xatol=1e-3, fatol=1e-8, maxiter=5000))
    return r.x


def angular_errors(Lt, K, up, yaw):
    """2D angle (deg) between each test segment and the image direction towards the nearer of the
    two horizontal vanishing points predicted by the camera."""
    d1, d2 = dirs(up, yaw)
    out = []
    for s in Lt:
        mid = (s[:2] + s[2:]) / 2; v = s[2:] - s[:2]; v /= np.linalg.norm(v)
        errs = []
        for d in (d1, d2):
            p = K @ d
            if abs(p[2]) < 1e-9:   # VP at infinity: direction is (p0,p1)
                t = p[:2]
            else:
                t = p[:2] / p[2] - mid
            t = t / (np.linalg.norm(t) + 1e-12)
            errs.append(np.degrees(np.arccos(np.clip(abs(v @ t), 0, 1))))
        out.append(min(errs))
    return np.array(out)


def height_metric(P, mask, K, up):
    ys, xs = np.nonzero(mask)
    Q = P[ys, xs]; ok = np.isfinite(Q).all(1); Q = Q[ok]; ys, xs = ys[ok], xs[ok]
    # distance along each pixel ray in the refined camera, matched to MoGe's metric distance |Q|
    rays = np.c_[xs + 0.5, ys + 0.5, np.ones(len(xs))] @ np.linalg.inv(K).T
    rays /= np.linalg.norm(rays, axis=1)[:, None]
    dist = np.linalg.norm(Q, axis=1)
    # floor point for height h along ray r: t = h / -(r.up)  ->  h = dist * -(r.up)
    hh = dist * -(rays @ up)
    return float(np.median(hh))


rows = []
for f in sorted(glob.glob(f'{D}/scenes/*.png')):
    n = os.path.basename(f)[:-4]
    bgr = cv2.imread(f); H, W = bgr.shape[:2]
    fm = cv2.imread(f'{D}/sam3/{n}_floor.png', 0) > 127
    o = moge_infer(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    P = o['points']; Kn = o['intrinsics']; K0 = Kn.copy(); K0[0] *= W; K0[1] *= H
    valid = o['mask'].astype(bool) & fm & np.isfinite(P).all(-1)
    valid &= cv2.erode(fm.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    pts = P[valid][::max(1, valid.sum() // 60000)]
    n0, h0, _ = fit_plane(pts)
    p0, r0 = up_to_pr(n0)
    hl0 = np.linalg.inv(K0).T @ n0
    hor0 = -(hl0[0] * W / 2 + hl0[2]) / hl0[1]
    f0 = K0[0, 0]
    L = line_segments(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    mid_y = (L[:, 1] + L[:, 3]) / 2
    ang = np.degrees(np.arctan2(L[:, 3] - L[:, 1], L[:, 2] - L[:, 0])) % 180
    is_vert = np.abs(ang - 90) < 8
    # test set: horizontal-ish segments below the horizon whose midpoint is on/next to the floor
    fm_d = cv2.dilate(fm.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
    mx = np.clip(((L[:, 0] + L[:, 2]) / 2).astype(int), 0, W - 1); my = np.clip(mid_y.astype(int), 0, H - 1)
    test = (~is_vert) & (mid_y > hor0 + 40) & fm_d[my, mx]
    train = ~test

    # (a) MoGe only (principal point at MoGe's, no line info): need a yaw -> take best yaw from TRAIN lines
    wtr = np.hypot(L[train, 2] - L[train, 0], L[train, 3] - L[train, 1])
    yaw_a = min(np.arange(0, 90, 0.25), key=lambda y: _mc(K0, p0, r0, y, L[train], wtr))
    err_a = angular_errors(L[test], K0, n0, yaw_a)
    # (b) refined with train lines only (held-out evaluation)
    xb = fit(L[train], W, H, f0, p0, r0, hor0)
    Kb = cam_K(xb[0], xb[1], W); upb = rot_up(xb[2], xb[3])
    err_b = angular_errors(L[test], Kb, upb, xb[4])
    # (c) final model: all lines
    xc = fit(L, W, H, f0, p0, r0, hor0)
    Kc = cam_K(xc[0], xc[1], W); upc = rot_up(xc[2], xc[3])
    hc = height_metric(P, valid, Kc, upc)
    hlc = np.linalg.inv(Kc).T @ upc
    res = dict(name=n, W=W, H=H, f_moge=float(f0), h_moge=float(h0), n_test=int(test.sum()), n_train=int(train.sum()),
               err_moge_med=float(np.median(err_a)) if len(err_a) else None,
               err_refined_heldout_med=float(np.median(err_b)) if len(err_b) else None,
               err_moge_p90=float(np.percentile(err_a, 90)) if len(err_a) else None,
               err_refined_heldout_p90=float(np.percentile(err_b, 90)) if len(err_b) else None,
               final=dict(f=float(xc[0]), cy=float(xc[1]), pitch=float(xc[2]), roll=float(xc[3]), yaw=float(xc[4]),
                          up=upc.tolist(), cam_h=hc, horizon_y=float(-(hlc[0] * W / 2 + hlc[2]) / hlc[1])))
    json.dump(res, open(f'{D}/geo2/{n}.json', 'w'), indent=1)
    rows.append(res)
    print(f"{n}: test lines {res['n_test']:3d} | angle err median MoGe {res['err_moge_med']:.2f}° -> refined(held-out) {res['err_refined_heldout_med']:.2f}° "
          f"| p90 {res['err_moge_p90']:.2f}° -> {res['err_refined_heldout_p90']:.2f}° | f {f0:.0f}->{xc[0]:.0f} ({(xc[0]/f0-1)*100:+.0f}%) "
          f"| cam h MoGe {h0:.2f} final {hc:.2f} m", flush=True)

json.dump(rows, open(f'{D}/geo2/summary.json', 'w'), indent=1)
