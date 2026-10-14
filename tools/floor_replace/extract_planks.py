"""Cut the scanned sample board into individual real planks (planks/p*.png).
Row seams / end joints were measured on '00 GRIS CHAUD' scan (3840x2423)."""
import cv2
t = cv2.imread('../../assets/00 GRIS CHAUD (RearLight) (300.00 dpi).jpg')
R = [68, 445, 828, 1203, 1582, 1962, 2344]          # horizontal seams (y)
ins = 14                                            # inset to drop the seam itself
spec = {0: [(100, 3690)], 1: [(100, 1878), (1902, 3690)], 2: [(100, 3690)],
        3: [(100, 1889), (1913, 3690)], 4: [(100, 3690)], 5: [(100, 3690)]}
n = 0
for r, segs in spec.items():
    for a, b in segs:
        cv2.imwrite(f'planks/p{n}.png', t[R[r] + ins:R[r + 1] - ins, a:b]); n += 1
print(n, 'planks written')
