"""Before/after for the user's circled mask flaws: old (SAM global + guided filter) vs new (tiled SAM + bg matting)."""
import sys, time, numpy as np, cv2
sys.path.insert(0, '/projects/sandbox/val')
from run_renders import cam, Hm, names, D
from floorkit import PlankLib, render
from cmp_mask import C

LIB = {'t1': PlankLib(f'{D}/lib/t1', 0.150, 1.212), 't3': PlankLib(f'{D}/lib/t3', 0.125, 0.909)}
JOBS = [(2, 't3'), (6, 't3'), (10, 't3'), (7, 't3'), (14, 't3'), (8, 't1'), (4, 't1'), (7, 't1')]
if len(sys.argv) > 1: JOBS = [j for j in JOBS if str(j[0]) in sys.argv[1:]]
for i, lk in JOBS:
    n = names[i - 1]; scene = cv2.imread(f'{D}/scenes/{n}.png')
    K, up, d1, d2, h = cam(n)
    e1, e2 = (d1, d2) if lk == 't1' else (d2, -d1)
    H = Hm(K, up, e1, e2, h)
    old_m = cv2.imread(f'{D}/sam3/{n}_floor.png', 0) > 127
    new_m = cv2.imread(f'{D}/sam3t/{n}_floor_prob.png', 0).astype(np.float32) / 255
    t = time.time()
    a = render(scene, old_m, H, LIB[lk], seed=i)
    b = render(scene, new_m, H, LIB[lk], seed=i)
    cv2.imwrite(f'{D}/renders/fix_{i:02d}_{lk}.jpg', b, [cv2.IMWRITE_JPEG_QUALITY, 93])
    for k, (X, Y, r) in enumerate(C[i]):
        y0, y1, x0, x1 = max(Y - r, 0), min(Y + r, 1536), max(X - r, 0), min(X + r, 2048)
        tl = [cv2.resize(im[y0:y1, x0:x1], (600, int(600 * (y1 - y0) / (x1 - x0))), interpolation=cv2.INTER_AREA)
              for im in (a, b)]
        for im, tag in zip(tl, ('before', 'after')):
            cv2.putText(im, tag, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6)
            cv2.putText(im, tag, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.imwrite(f'{D}/fixcmp_{i:02d}_{lk}_{k}.jpg', np.hstack([tl[0], np.full((tl[0].shape[0], 8, 3), 255, np.uint8), tl[1]]))
    print(i, lk, f'{time.time()-t:.1f}s', flush=True)
