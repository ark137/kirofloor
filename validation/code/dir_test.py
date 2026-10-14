"""Real-size re-render + laying-direction comparison.
For each scene: floor extent (m) along the two Manhattan wall directions, render t1 both ways."""
import sys, json, time, numpy as np, cv2
sys.path.insert(0, '/projects/sandbox/val')
from run_renders import cam, Hm, names, D  # noqa (run_renders guarded below)
from floorkit import PlankLib, render

t1 = PlankLib(f'{D}/lib/t1', 0.150, 1.212)
t3 = PlankLib(f'{D}/lib/t3', 0.125, 0.909)
print('t1 x_scale', round(t1.x_scale, 3), 'lengths', np.round(t1.len, 3))
print('t3 x_scale', round(t3.x_scale, 3), 'lengths', np.round(t3.len[:3], 3))


def extent(n):
    K, up, d1, d2, h = cam(n)
    fm = cv2.imread(f'{D}/sam3/{n}_floor.png', 0) > 127
    ys, xs = np.nonzero(fm[::4, ::4]); P = np.c_[xs * 4, ys * 4, np.ones(len(xs))].T
    Q = np.linalg.inv(Hm(K, up, d1, d2, h)) @ P; Q = Q[:2] / Q[2]
    ok = np.isfinite(Q).all(0) & (np.abs(Q).max(0) < 30)
    lo, hi = np.percentile(Q[:, ok], [3, 97], axis=1)
    return hi - lo   # extent along d1, d2


sel = [int(a) for a in sys.argv[1:]] or [1, 4, 8]
info = {}
for i in range(1, 15):
    e = extent(names[i - 1]); info[i] = e
    print(f'scene {i:2d}: extent d1 {e[0]:.2f} m  d2 {e[1]:.2f} m  -> longer: {"d1" if e[0] > e[1] else "d2"}')
for i in sel:
    n = names[i - 1]; scene = cv2.imread(f'{D}/scenes/{n}.png')
    fm = cv2.imread(f'{D}/sam3/{n}_floor.png', 0) > 127
    K, up, d1, d2, h = cam(n); outs = []
    for tag, (e1, e2) in (('d1', (d1, d2)), ('d2', (d2, -d1))):
        t = time.time(); o = render(scene, fm, Hm(K, up, e1, e2, h), t1, layout='straight', seed=i)
        cv2.imwrite(f'{D}/renders/dir_{i:02d}_{tag}.jpg', o, [cv2.IMWRITE_JPEG_QUALITY, 93]); outs.append(o)
        print(i, tag, f'{time.time()-t:.1f}s')
    sm = [cv2.resize(o, (1024, 768), interpolation=cv2.INTER_AREA) for o in outs]
    for s, tag in zip(sm, ('A', 'B')):
        cv2.putText(s, tag, (24, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 9)
        cv2.putText(s, tag, (24, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 3)
    cv2.imwrite(f'{D}/renders/dir_cmp_{i:02d}.jpg', np.hstack(sm), [cv2.IMWRITE_JPEG_QUALITY, 90])
json.dump({k: list(map(float, v)) for k, v in info.items()}, open(f'{D}/extent.json', 'w'))
