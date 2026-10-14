"""Cut the scanned sample board into individual real planks (planks/p*.png).

The scan embeds a scanner ICC profile (Concord_Rear_ICC). Viewers like Photoshop apply it,
so the *true* colour is the ICC-converted one. We convert to sRGB (relative colorimetric +
black point compensation) before cutting, otherwise the floor comes out greyer/cooler.
Row seams / end joints were measured on '00 GRIS CHAUD' scan (3840x2423)."""
import io
import numpy as np, cv2
from PIL import Image, ImageCms

SCAN = '../../assets/00 GRIS CHAUD (RearLight) (300.00 dpi).jpg'
im = Image.open(SCAN).convert('RGB')
icc = Image.open(SCAN).info.get('icc_profile')
if icc:
    src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
    print('ICC:', ImageCms.getProfileDescription(src).strip(), '-> sRGB')
    tr = ImageCms.buildTransform(src, ImageCms.createProfile('sRGB'), 'RGB', 'RGB',
                                 renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                                 flags=ImageCms.Flags.BLACKPOINTCOMPENSATION)
    im = ImageCms.applyTransform(im, tr)
else:
    print('no ICC profile, assuming sRGB')
t = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)

R = [68, 445, 828, 1203, 1582, 1962, 2344]          # horizontal seams (y)
ins = 14                                            # inset to drop the seam itself
spec = {0: [(100, 3690)], 1: [(100, 1878), (1902, 3690)], 2: [(100, 3690)],
        3: [(100, 1889), (1913, 3690)], 4: [(100, 3690)], 5: [(100, 3690)]}
n = 0
for r, segs in spec.items():
    for a, b in segs:
        cv2.imwrite(f'planks/p{n}.png', t[R[r] + ins:R[r + 1] - ins, a:b]); n += 1
print(n, 'planks written (sRGB)')
