"""SAM 3 'floor' at higher effective resolution: global pass + 3x3 overlapping tiles, soft probabilities.
Global mask cell = 7 px at 2048 wide; tiles 768 px -> 2.7 px. Output: sam3t/{name}_floor_prob.png (0-255)."""
import sys, time, glob, os
import numpy as np, cv2, torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

torch.set_num_threads(8)
MID = glob.glob('/projects/sandbox/.hf/hub/models--jetjodh--sam3/snapshots/*')[0]
proc = Sam3Processor.from_pretrained(MID); model = Sam3Model.from_pretrained(MID).eval()
D = '/projects/sandbox/val'; os.makedirs(f'{D}/sam3t', exist_ok=True)
names = sorted(os.path.basename(f)[:-4] for f in glob.glob(f'{D}/scenes/*.png'))


def prob_map(img, thr):
    """img: PIL RGB -> soft floor probability at img resolution."""
    w, h = img.size
    inp = proc(images=img, text='floor', return_tensors='pt')
    with torch.inference_mode():
        o = model(**inp)
    sc = torch.sigmoid(o.pred_logits[0]) * torch.sigmoid(o.presence_logits[0, 0])
    m = torch.sigmoid(o.pred_masks[0])                     # 200 x 288 x 288
    keep = sc > thr
    p = (m[keep].max(0).values if keep.any() else torch.zeros_like(m[0])).numpy()
    return cv2.resize(p.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 1)


for i in [int(a) for a in sys.argv[1:]]:
    n = names[i - 1]; t = time.time()
    img = Image.open(f'{D}/scenes/{n}.png').convert('RGB'); W, H = img.size
    g = prob_map(img, 0.4)
    T = 768; xs = np.linspace(0, W - T, 3).astype(int); ys = np.linspace(0, H - T, 3).astype(int)
    win = np.outer(np.hanning(T + 2)[1:-1], np.hanning(T + 2)[1:-1]).astype(np.float32) + 1e-3
    acc = np.zeros((H, W), np.float32); wsum = np.zeros((H, W), np.float32)
    gate = cv2.dilate((g > 0.5).astype(np.uint8), np.ones((97, 97), np.uint8)) > 0
    for y in ys:
        for x in xs:
            if not gate[y:y + T, x:x + T].any(): continue
            p = prob_map(img.crop((x, y, x + T, y + T)), 0.3)
            acc[y:y + T, x:x + T] += p * win; wsum[y:y + T, x:x + T] += win
    tp = np.where(wsum > 0, acc / np.maximum(wsum, 1e-6), g)
    fused = np.where(gate, tp, 0)
    cv2.imwrite(f'{D}/sam3t/{n}_floor_prob.png', (fused * 255).astype(np.uint8))
    cv2.imwrite(f'{D}/sam3t/{n}_global_prob.png', (g * 255).astype(np.uint8))
    print(i, n, f'{time.time()-t:.0f}s', 'area global', round(float((g > .5).mean()), 4),
          'tiled', round(float((fused > .5).mean()), 4), flush=True)
