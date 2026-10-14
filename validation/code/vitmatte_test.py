"""Try ViTMatte for wide/blurred edges: trimap from tiled-SAM prob, alpha from ViTMatte, composite with known bg."""
import sys, os, glob, time, numpy as np, cv2, torch
from transformers import VitMatteImageProcessor, VitMatteForImageMatting
from PIL import Image
torch.set_num_threads(8)
D = '/projects/sandbox/val'
names = sorted(os.path.basename(f)[:-4] for f in glob.glob(f'{D}/scenes/*.png'))
MID = 'hustvl/vitmatte-small-composition-1k'
proc = VitMatteImageProcessor.from_pretrained(MID); model = VitMatteForImageMatting.from_pretrained(MID).eval()
os.makedirs(f'{D}/matte', exist_ok=True)
for i in [int(a) for a in sys.argv[1:]]:
    n = names[i - 1]; t = time.time()
    sc = cv2.imread(f'{D}/scenes/{n}.png'); H, W = sc.shape[:2]
    p = cv2.imread(f'{D}/sam3t/{n}_floor_prob.png', 0) > 127
    k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    sf = cv2.erode(p.astype(np.uint8), k(6)) > 0          # sure floor
    sb = cv2.dilate(p.astype(np.uint8), k(int(os.environ.get("RO", 16)))) == 0       # sure object
    tri = np.full((H, W), 128, np.uint8); tri[sf] = 0; tri[sb] = 255     # matte = OBJECT alpha
    # run at 1024 wide (ViTMatte is heavy), upsample alpha
    s = 1024 / W; w2, h2 = 1024, round(H * s)
    img = Image.fromarray(cv2.cvtColor(cv2.resize(sc, (w2, h2), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    tr = Image.fromarray(cv2.resize(tri, (w2, h2), interpolation=cv2.INTER_NEAREST))
    inp = proc(images=img, trimaps=tr, return_tensors='pt')
    with torch.inference_mode():
        a = model(**inp).alphas[0, 0].numpy()[:h2, :w2]
    a = cv2.resize(a, (W, H), interpolation=cv2.INTER_CUBIC).clip(0, 1)
    fa = np.where(sf, 1.0, np.where(sb, 0.0, 1 - a)).astype(np.float32)     # floor alpha
    cv2.imwrite(f'{D}/matte/{n}_floor_alpha.png', (fa * 255).astype(np.uint8))
    print(i, f'{time.time()-t:.0f}s', flush=True)
