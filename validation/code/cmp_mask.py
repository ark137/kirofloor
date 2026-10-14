import sys, glob, os, numpy as np, cv2
D = '/projects/sandbox/val'
names = sorted(os.path.basename(f)[:-4] for f in glob.glob(f'{D}/scenes/*.png'))
# scene -> (cx, cy, r) in full-res px
C = {2: [(1436, 957, 610), (1883, 970, 425)], 6: [(400, 900, 560)], 10: [(1862, 1170, 690)],
     7: [(1782, 1197, 800)], 14: [(1596, 1250, 800)], 8: [(306, 904, 425), (638, 944, 610)], 4: [(1277, 1117, 690)]}
if __name__ == '__main__':
    for i in [int(a) for a in sys.argv[1:]]:
        n = names[i - 1]; sc = cv2.imread(f'{D}/scenes/{n}.png')
        old = cv2.imread(f'{D}/sam3/{n}_floor.png', 0) > 127
        new = cv2.imread(f'{D}/sam3t/{n}_floor_prob.png', 0) > 127
        for k, (X, Y, r) in enumerate(C[i]):
            y0, y1, x0, x1 = max(Y - r, 0), min(Y + r, 1536), max(X - r, 0), min(X + r, 2048)
            t = []
            for m in (old, new):
                ov = sc.copy(); ov[m] = (0.5 * ov[m] + [0, 0, 127]).astype(np.uint8)
                t.append(cv2.resize(ov[y0:y1, x0:x1], (520, int(520 * (y1 - y0) / (x1 - x0)))))
            cv2.imwrite(f'{D}/mcmp_{i:02d}_{k}.jpg', np.hstack(t))
