"""SAM 3 text-prompted segmentation: floor + floor-covering objects (rugs)."""
import sys, time, glob, os, json
import numpy as np, cv2, torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

torch.set_num_threads(8)
MID = glob.glob('/projects/sandbox/.hf/hub/models--jetjodh--sam3/snapshots/*')[0]
t0 = time.time()
proc = Sam3Processor.from_pretrained(MID)
model = Sam3Model.from_pretrained(MID).eval()
print(f'load {time.time()-t0:.1f}s')

PROMPTS = ['floor', 'rug']
os.makedirs('/projects/sandbox/val/sam3', exist_ok=True)
files = sys.argv[1:] or sorted(glob.glob('/projects/sandbox/val/scenes/*.png'))
for f in files:
    name = os.path.basename(f)[:-4]
    img = Image.open(f).convert('RGB')
    W, H = img.size
    out = {}
    t = time.time()
    for p in PROMPTS:
        inp = proc(images=img, text=p, return_tensors='pt')
        with torch.inference_mode():
            o = model(**inp)
        r = proc.post_process_instance_segmentation(o, threshold=0.4, mask_threshold=0.5,
                                                    target_sizes=inp.get('original_sizes').tolist())[0]
        masks = r['masks'].cpu().numpy().astype(bool) if len(r['masks']) else np.zeros((0, H, W), bool)
        scores = r['scores'].cpu().numpy() if len(r['scores']) else np.zeros(0)
        union = masks.any(0) if len(masks) else np.zeros((H, W), bool)
        cv2.imwrite(f'/projects/sandbox/val/sam3/{name}_{p}.png', union.astype(np.uint8) * 255)
        out[p] = dict(n=int(len(masks)), scores=[round(float(s), 3) for s in scores], area=round(float(union.mean()), 4))
    print(name, f'{time.time()-t:.1f}s', json.dumps(out), flush=True)
