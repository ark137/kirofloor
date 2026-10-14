import cv2, numpy as np
im = cv2.imread('../../assets/gemini_1772203361325.png')
N = 1024
s = cv2.resize(im, (N, N), interpolation=cv2.INTER_AREA)
k = N / 1200  # coords below are in 1200px preview space

VP1 = np.array([1605.4, 422.7]); VP2 = np.array([-727.7, 419.4])
def line_y(p, vp, x): return p[1] + (vp[1] - p[1]) * (x - p[0]) / (vp[0] - p[0])

xs = np.arange(N) / k
corner = np.array([850, 539])
ytop = np.where(xs < 850, line_y(corner, VP1, xs), line_y(corner, VP2, xs))
ytop = np.where(xs > 1136, 646 + (xs - 1136) * 0.11, ytop)  # pillar base
yy = np.arange(N)[:, None] / k
xx = np.arange(N)[None, :] / k
below = yy > ytop[None, :] + 1

m = np.full((N, N), cv2.GC_BGD, np.uint8)
m[below] = cv2.GC_PR_FGD
sure = (yy > 820) | ((yy > 700) & (xx > 520)) | ((yy > 620) & (xx > 800) & (xx < 1120))
m[sure & below] = cv2.GC_FGD

def poly(p, v):
    cv2.fillPoly(m, [(np.array(p) * k).astype(np.int32)], int(v))
B, F = cv2.GC_BGD, cv2.GC_FGD
poly([(0, 520), (150, 520), (150, 735), (75, 770), (0, 735)], B)                      # arm
poly([(160, 600), (495, 600), (485, 705), (240, 760), (165, 740)], B)                 # chaise
poly([(160, 515), (660, 515), (655, 640), (500, 672), (495, 600), (160, 600)], B)     # back + right seat
poly([(670, 575), (770, 575), (770, 625), (670, 625)], B)                              # table
poly([(915, 480), (990, 480), (990, 555), (915, 555)], B)                              # pot
poly([(640, 520), (680, 520), (680, 565), (640, 565)], B)                              # vase
# contact-shadow strip just below the sofa = floor
poly([(0, 775), (75, 805), (150, 778), (240, 795), (500, 728), (640, 690), (660, 690),
      (660, 720), (500, 750), (240, 820), (0, 820)], F)

bgm = np.zeros((1, 65)); fgm_ = np.zeros((1, 65))
cv2.grabCut(s, m, None, bgm, fgm_, 8, cv2.GC_INIT_WITH_MASK)
fg = np.isin(m, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8)
n, lab, st, _ = cv2.connectedComponentsWithStats(fg)
fg = (lab == 1 + np.argmax(st[1:, 4])).astype(np.uint8)
fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

# --- explicit fix for the bright floor around the coffee table (GrabCut confuses it with wall)
fix = np.zeros_like(fg)
def fpoly(img, p, v): cv2.fillPoly(img, [(np.array(p) * k).astype(np.int32)], int(v))
fpoly(fix, [(560, 564), (628, 560), (754, 548), (778, 542), (860, 540), (860, 700), (560, 700)], 1)
# ---- traced table + vase polygons (crop coords from 6.11x zoom grid, origin 620,505)
def c2p(pts): return [(620 + x / 6.11, 505 + y / 6.11) for x, y in pts]
table = c2p([(285, 380), (960, 433), (960, 497), (667, 554), (667, 720), (365, 778), (285, 765)])
vase = c2p([(195, 80), (295, 80), (297, 190), (330, 220), (354, 290), (332, 355), (300, 388),
            (240, 402), (170, 362), (133, 290), (150, 220), (193, 186)])
top = c2p([(0, 355), (795, 262), (830, 262), (965, 272)]) + [(860, 540)]
FS = 4096
kk = FS / 1200
def P(p): return (np.array(p) * kk).astype(np.int32)
big = (cv2.resize(fg.astype(np.float32), (FS, FS), interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)
fixb = np.zeros_like(big)
cv2.fillPoly(fixb, [P(top + [(860, 660), (560, 660), (560, 565)])], 1)
cv2.fillPoly(fixb, [P(table)], 0, lineType=cv2.LINE_8)
cv2.fillPoly(fixb, [P(vase)], 0)
sofa = np.zeros_like(big); cv2.fillPoly(sofa, [P([(0, 565), (666.6, 565), (666.6, 645), (500, 680), (0, 780)])], 1)
zone = np.zeros_like(big); cv2.fillPoly(zone, [P([(668, 500), (860, 500), (860, 660), (668, 660)])], 1)
big = np.where((zone == 1) & (sofa == 0), fixb, big).astype(np.uint8)
# ---- back wall base + pot + pillar (traced from zoom grids)
fix2 = np.zeros_like(big)
cv2.fillPoly(fix2, [P([(840, 538.0), (900, 541.5), (1000, 549.5), (1136, 559), (1136, 644), (1200, 650.5),
                       (1200, 720), (840, 720)])], 1)
cv2.fillPoly(fix2, [P([(909, 515), (995, 515), (995, 548), (990, 556), (980, 561), (965, 563.5), (935, 563.5),
                       (922, 560), (914, 553), (910, 545)])], 0)
zone2 = np.zeros_like(big); cv2.fillPoly(zone2, [P([(840, 500), (1200, 500), (1200, 720), (840, 720)])], 1)
big = np.where(zone2 == 1, fix2, big).astype(np.uint8)
cv2.imwrite('mask4096.png', big * 255)
fg = cv2.resize(big, (N, N), interpolation=cv2.INTER_AREA)
cv2.imwrite('mask1024.png', fg * 255)
vis = s.copy(); vis[fg == 0] = (vis[fg == 0] * 0.35).astype(np.uint8)
cv2.imwrite('mask_vis2.jpg', vis)
