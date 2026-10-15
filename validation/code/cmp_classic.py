"""Compare model-free mask vs SAM 3 reference. Overlay: green = both, red = extra (classic only), blue = missed."""
import sys, os, glob, json, numpy as np, cv2
D = '/projects/sandbox/val'
names = sorted(os.path.basename(f)[:-4] for f in glob.glob(f'{D}/scenes/*.png'))


def ref_mask(n):
    f = f'{D}/sam3t/{n}_floor_prob.png'
    return (cv2.imread(f, 0) > 127) if os.path.exists(f) else (cv2.imread(f'{D}/sam3/{n}_floor.png', 0) > 127)


def overlay(sc, c, r):
    o = sc.astype(np.float32) * 0.55
    o[c & r] += np.array([0, 110, 0]); o[c & ~r] += np.array([0, 0, 200]); o[~c & r] += np.array([200, 60, 0])
    return np.clip(o, 0, 255).astype(np.uint8)


if __name__ == '__main__':
    tag = sys.argv[1] if len(sys.argv) > 1 else 'classic'
    res = {}; tiles = []
    for i in range(1, 15):
        n = names[i - 1]; f = f'{D}/{tag}/{n}_floor.png'
        if not os.path.exists(f): continue
        c = cv2.imread(f, 0) > 127; r = ref_mask(n); sc = cv2.imread(f'{D}/scenes/{n}.png')
        iou = (c & r).sum() / max((c | r).sum(), 1)
        extra = (c & ~r).sum() / r.sum(); miss = (~c & r).sum() / r.sum()
        res[i] = dict(iou=round(float(iou), 3), extra=round(float(extra), 3), miss=round(float(miss), 3))
        t = cv2.resize(overlay(sc, c, r), (512, 384), interpolation=cv2.INTER_AREA)
        cv2.putText(t, f'{i}  IoU {iou:.2f}', (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(t, f'{i}  IoU {iou:.2f}', (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        tiles.append(t)
        print(i, res[i], flush=True)
    while len(tiles) % 4: tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)])
    cv2.imwrite(f'{D}/{tag}_grid.jpg', grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    json.dump(res, open(f'{D}/{tag}_scores.json', 'w'), indent=1)
