import os, glob, json, sys, time
import numpy as np, cv2
sys.path.insert(0, '/projects/sandbox/val')
from floorkit import PlankLib, render

D = '/projects/sandbox/val'
os.makedirs(f'{D}/renders', exist_ok=True)
names = sorted(os.path.basename(f)[:-4] for f in glob.glob(f'{D}/scenes/*.png'))


def cam(n):
    r = json.load(open(f'{D}/geo2/{n}.json'))['final']
    W = 2048
    K = np.array([[r['f'], 0, W / 2], [0, r['f'], r['cy']], [0, 0, 1.0]])
    up = np.array(r['up'])
    a = np.cross(up, [1, 0, 0]); a /= np.linalg.norm(a); b = np.cross(up, a)
    t = np.radians(r['yaw']); d1 = np.cos(t) * a + np.sin(t) * b; d2 = np.cross(up, d1)
    return K, up, d1, d2, r['cam_h']


def Hm(K, up, e1, e2, h):
    return K @ np.c_[e1, e2, -h * up]


if __name__ == '__main__':
    LIBS = {
        't1': PlankLib(f'{D}/lib/t1', 0.150, 1.212),  # GRIS CHAUD 150x1212 (scan is x-stretched, corrected)
        't3': PlankLib(f'{D}/lib/t3', 0.125, 0.909),  # single boards 125x909
        't2': PlankLib(f'{D}/lib/t2', 0.09),         # herringbone 90 x ~600
    }
    for k, l in LIBS.items():
        print(k, 'pieces', len(l), 'length m', np.round(l.len, 3)[:4], 'median L*', round(l.median_L * 100 / 255, 1))

    # (scene index 1-based, library, layout, direction 'd1'/'d2')
    JOBS = [(i, 't1', 'straight', 'd1') for i in range(1, 15)] + \
           [(i, 't3', 'straight', 'd2') for i in (2, 6, 7, 10, 14)] + \
           [(i, 't2', 'herringbone', 'd1') for i in (3, 5, 9, 12, 13)]
    if len(sys.argv) > 1: JOBS = [j for j in JOBS if str(j[0]) in sys.argv[1:]]
    for (i, lk, lay, dr) in JOBS:
        n = names[i - 1]
        scene = cv2.imread(f'{D}/scenes/{n}.png')
        fm = cv2.imread(f'{D}/sam3/{n}_floor.png', 0) > 127
        K, up, d1, d2, h = cam(n)
        e1, e2 = (d1, d2) if dr == 'd1' else (d2, -d1)
        t = time.time()
        out = render(scene, fm, Hm(K, up, e1, e2, h), LIBS[lk], layout=lay, seed=i)
        fn = f'{D}/renders/{i:02d}_{lk}_{lay}.jpg'
        cv2.imwrite(fn, out, [cv2.IMWRITE_JPEG_QUALITY, 93])
        print(f'{i:2d} {lk} {lay:11s} {time.time()-t:.1f}s -> {os.path.basename(fn)}', flush=True)
