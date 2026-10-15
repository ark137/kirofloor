"""Model-free floor mask for grey-floor AI scenes (no neural nets).
Cues: floor = low-chroma, low-texture, connected to the lower part of the image, bounded by edges.
1) floor colour model from the lower image (robust chroma median)
2) per-pixel floor likelihood (chroma distance, local texture), L allowed to vary (sun/shadow)
3) GrabCut seeded with sure-floor (bottom, high likelihood) / sure-bg (top band, low likelihood)
4) keep components connected to the seeds; remove specks
Output: classic/{name}_floor.png (0/255, full res)."""
import sys, os, glob, time, json
import numpy as np, cv2

D = '/projects/sandbox/val'
MODE = os.environ.get('MODE', 'gc')
OUT = 'classic' if MODE == 'gc' else 'classic_ws'
os.makedirs(f'{D}/{OUT}', exist_ok=True)
names = sorted(os.path.basename(f)[:-4] for f in glob.glob(f'{D}/scenes/*.png'))


def floor_mask(bgr, W2=1024):
    H0, W0 = bgr.shape[:2]
    s = W2 / W0; img = cv2.resize(bgr, (W2, round(H0 * s)), interpolation=cv2.INTER_AREA)
    H, W = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = lab[..., 0] * 100 / 255, lab[..., 1] - 128, lab[..., 2] - 128
    # local texture: std of L at small scale, relative to brightness (shadows keep low texture)
    m = cv2.blur(L, (5, 5)); tex = np.sqrt(np.maximum(cv2.blur(L * L, (5, 5)) - m * m, 0))   # absolute L std
    tex = cv2.blur(tex, (5, 5))
    rows = np.arange(H)[:, None] / H
    # brightness-normalised chroma: shadows keep the same ratio, dark wood/fabric does not
    na, nb = a / (L + 8) * 50, b / (L + 8) * 50
    C = np.hypot(a, b)
    cand = (rows > 0.55) & (tex < 0.8) & (C < 30)
    if cand.sum() < 500: cand = (rows > 0.55) & (tex < np.percentile(tex[rows[:, 0] > 0.55], 30))
    a0, b0 = np.median(na[cand]), np.median(nb[cand])
    da = np.hypot(na - a0, nb - b0)
    sig = max(2.5, np.percentile(da[cand], 85))
    like = np.exp(-(da / ((2.2 if MODE == 'ws' else 1.5) * sig)) ** 2) * np.exp(-(tex / (1.8 if MODE == 'ws' else 1.2)) ** 2)
    like = cv2.GaussianBlur(like, (0, 0), 1.5)
    # GrabCut on a feature image (L, chroma ratio, texture) so it separates smooth floor from fabric
    feat = np.dstack([np.clip(L * 2.55, 0, 255), np.clip(128 + 4 * (da - sig), 0, 255),
                      np.clip(tex * 60, 0, 255)]).astype(np.uint8)
    if MODE == 'ws':
        # watershed: grow sure-floor and sure-object markers until they meet on the strongest edges
        mk = np.zeros((H, W), np.int32)
        sf = (like > 0.5) & (rows > 0.5)
        sf = cv2.erode(sf.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        n, cc, st, _ = cv2.connectedComponentsWithStats(sf.astype(np.uint8))
        big = np.zeros(n, bool); big[1:] = st[1:, 4] > 0.002 * H * W; sf = big[cc]
        Lf = np.median(L[sf]) if sf.any() else 60
        sb = (like < 0.01) | (tex > 4.0) | (L < 0.4 * Lf)
        sb = cv2.erode(sb.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        sb |= (rows < 0.3).repeat(W, 1)
        mk[sb] = 2; mk[sf] = 1
        ws = cv2.watershed(cv2.GaussianBlur(img, (0, 0), 1.0), mk)
        fg = ws == 1
        fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
        out = cv2.resize(fg.astype(np.uint8) * 255, (W0, H0), interpolation=cv2.INTER_LINEAR) > 127
        return out, dict(a0=round(float(a0), 1), b0=round(float(b0), 1), sig=round(float(sig), 1))
    # GrabCut init
    gc = np.full((H, W), cv2.GC_PR_BGD, np.uint8)
    gc[like > 0.35] = cv2.GC_PR_FGD
    sure_f = (like > 0.6) & (rows > 0.55)
    sure_f = cv2.erode(sure_f.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    gc[sure_f] = cv2.GC_FGD
    gc[(rows < 0.30).repeat(W, 1)] = cv2.GC_BGD
    gc[(like < 0.03) & (rows < 0.6)] = cv2.GC_BGD
    bgm = np.zeros((1, 65), np.float64); fgm = np.zeros((1, 65), np.float64)
    cv2.grabCut(feat, gc, None, bgm, fgm, 5, cv2.GC_INIT_WITH_MASK)
    fg = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
    # keep components touching sure-floor seeds; drop specks
    n, lab_cc, st, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=4)
    seed_ids = np.unique(lab_cc[sure_f & fg]); keep = np.zeros(n, bool); keep[seed_ids] = True; keep[0] = False
    keep &= st[:, 4] > 0.0005 * H * W
    fg = keep[lab_cc]
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    out = cv2.resize(fg.astype(np.uint8) * 255, (W0, H0), interpolation=cv2.INTER_LINEAR) > 127
    info = dict(a0=round(float(a0), 1), b0=round(float(b0), 1), sig=round(float(sig), 1))
    return out, info


if __name__ == '__main__':
    for i in [int(x) for x in sys.argv[1:]] or range(1, 15):
        n = names[i - 1]; t = time.time()
        m, info = floor_mask(cv2.imread(f'{D}/scenes/{n}.png'))
        cv2.imwrite(f'{D}/{OUT}/{n}_floor.png', m.astype(np.uint8) * 255)
        print(f'{i:2d} {time.time()-t:5.1f}s area {m.mean():.3f} {json.dumps(info)}', flush=True)
