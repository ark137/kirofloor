"""Type 2: several boards photographed together on a rack, white background.
Find each board, fit its 4 edges, rectify to a rectangle, trim tongue/bevel, remove the
in-board lighting gradient (keep board-to-board differences)."""
import glob, os
import numpy as np, cv2

OUT = '/projects/sandbox/val/lib/t2'
os.makedirs(OUT, exist_ok=True)
S = 4  # analysis downscale


def board_mask(small):
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0] * 100 / 255; b = lab[..., 2] - 128
    m = (L > 55) & (L < 89) & (b < 12)
    return cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


def runs(v, thr):
    on = v > thr; out = []; i = 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]: j += 1
            out.append((i, j)); i = j
        else: i += 1
    return out


def fit_edge(pts):
    """robust line x = a*y + c through edge points (y, x)."""
    y, x = pts[:, 0].astype(float), pts[:, 1].astype(float)
    keep = np.ones(len(y), bool)
    for _ in range(4):
        A = np.c_[y[keep], np.ones(keep.sum())]
        a, c = np.linalg.lstsq(A, x[keep], rcond=None)[0]
        r = np.abs(x - (a * y + c)); keep = r < max(2.0, 2.5 * np.median(r[keep]))
    return a, c


n = 0
for f in [p for p in sorted(glob.glob('/projects/sandbox/work/t2/017A*.jpg')) if '_s' not in p]:
    if '_s' in f: continue
    im = cv2.imread(f); H, W = im.shape[:2]
    sm = cv2.resize(im, (W // S, H // S), interpolation=cv2.INTER_AREA)
    m = board_mask(sm); h, w = m.shape
    band = m[int(h * .25):int(h * .75)]
    col = band.mean(0)
    for (x0, x1) in runs(col, 0.6):
        if x1 - x0 < w * 0.03: continue
        sub = m[:, x0:x1]
        rowf = sub.mean(1); rr = [r for r in runs(rowf, 0.5) if r[1] - r[0] > h * 0.4]
        if not rr: continue
        y0, y1 = max(rr, key=lambda r: r[1] - r[0])
        # left / right edge points per row (full-res coordinates)
        ys = np.arange(y0 + 10, y1 - 10, 2)
        lp, rp = [], []
        for y in ys:
            on = np.where(m[y, max(0, x0 - 10):min(w, x1 + 10)])[0]
            if len(on) < 5: continue
            lp.append((y, on[0] + max(0, x0 - 10))); rp.append((y, on[-1] + max(0, x0 - 10)))
        la, lc = fit_edge(np.array(lp)); ra, rc = fit_edge(np.array(rp))
        # top / bottom edges: rows where the board starts/ends inside the central columns
        xc = np.arange(int(x0 + (x1 - x0) * .2), int(x0 + (x1 - x0) * .8))
        tops, bots = [], []
        for x in xc:
            on = np.where(m[:, x])[0]; on = on[(on > y0 - 20) & (on < y1 + 20)]
            if len(on): tops.append((x, on[0])); bots.append((x, on[-1]))
        def fit_h(p):
            p = np.array(p, float); A = np.c_[p[:, 0], np.ones(len(p))]
            return np.linalg.lstsq(A, p[:, 1], rcond=None)[0]
        ta, tc = fit_h(tops); ba, bc = fit_h(bots)
        def inter(xa, xcst, ya, ycst):   # x = xa*y + xcst ; y = ya*x + ycst
            y = (ya * xcst + ycst) / (1 - ya * xa); x = xa * y + xcst; return np.array([x, y]) * S
        TL, TR = inter(la, lc, ta, tc), inter(ra, rc, ta, tc)
        BL, BR = inter(la, lc, ba, bc), inter(ra, rc, ba, bc)
        wpx = (np.linalg.norm(TR - TL) + np.linalg.norm(BR - BL)) / 2
        hpx = (np.linalg.norm(BL - TL) + np.linalg.norm(BR - TR)) / 2
        Wt, Ht = int(round(wpx)), int(round(hpx))
        Mw = cv2.getPerspectiveTransform(np.float32([TL, TR, BR, BL]), np.float32([[0, 0], [Wt, 0], [Wt, Ht], [0, Ht]]))
        board = cv2.warpPerspective(im, Mw, (Wt, Ht), flags=cv2.INTER_LANCZOS4)
        # trim bevel / tongue / rounded corners
        ix, iy = int(Wt * 0.03), int(Wt * 0.03)
        board = board[iy:Ht - iy, ix:Wt - ix]
        board = cv2.rotate(board, cv2.ROTATE_90_CLOCKWISE)          # length horizontal
        # in-board lighting: fit a smooth (bilinear+quadratic along length) gain in log-luminance
        lab = cv2.cvtColor(board, cv2.COLOR_BGR2LAB).astype(np.float32)
        Lc = lab[..., 0]
        hh, ww = Lc.shape
        yy, xx = np.mgrid[0:hh:8, 0:ww:8]
        v = np.log(cv2.GaussianBlur(Lc, (0, 0), 25)[0:hh:8, 0:ww:8] + 1)
        u = xx / ww - .5; t = yy / hh - .5
        A = np.c_[u.ravel(), t.ravel(), (u ** 2).ravel(), (u * t).ravel()]
        coef = np.linalg.lstsq(A, (v - v.mean()).ravel(), rcond=None)[0]
        Yf, Xf = np.mgrid[0:hh, 0:ww]; U = Xf / ww - .5; T = Yf / hh - .5
        g = np.exp(coef[0] * U + coef[1] * T + coef[2] * (U ** 2 - 1 / 12) + coef[3] * U * T)
        corr = board.astype(np.float32) / g[..., None]
        cv2.imwrite(f'{OUT}/p{n}.png', np.clip(corr, 0, 255).astype(np.uint8))
        cv2.imwrite(f'{OUT}/raw{n}.png', board)
        span = (np.exp(np.abs(coef[0]) * .5 + np.abs(coef[2]) * .25) - 1) * 100
        print(f'{os.path.basename(f)} board {n}: {board.shape[1]}x{board.shape[0]} px, ratio {board.shape[1]/board.shape[0]:.2f}, '
              f'skew L/R {np.degrees(np.arctan(la)):.2f}/{np.degrees(np.arctan(ra)):.2f} deg, gradient removed ~{span:.1f}%')
        n += 1
print('boards:', n)
