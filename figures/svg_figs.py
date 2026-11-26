"""Professional SVG versions of Fig. 1 (framework) and Fig. 3 (host network). Rendered with Playwright."""
import math, random

INK = "#1d1d1b"; INK2 = "#55544f"; INK3 = "#8a8983"; LINE = "#b9b7ae"
BLUE = "#2a78d6"; BLUE_L = "#e3eefb"; BLUE_M = "#9ec5f4"
ORG = "#eb6834"; ORG_L = "#fdeee6"; ORG_M = "#f6b796"
VIO = "#6250d6"; VIO_L = "#ecebfb"; VIO_M = "#b9b1f0"
RED = "#e34948"; RED_L = "#fcebeb"
GRN = "#008300"; GRN_L = "#e5f4e4"
GRY_L = "#f4f3ef"

FONT = "Inter, 'DejaVu Sans', sans-serif"


class SVG:
    def __init__(self, w, h):
        self.w, self.h, self.e = w, h, []

    def add(self, s):
        self.e.append(s)

    def rect(self, x, y, w, h, fill="none", stroke=LINE, sw=1.2, rx=10, dash=None, op=1):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d} opacity="{op}"/>')

    def text(self, x, y, s, size=13, weight=400, fill=INK, anchor="middle", style="", ls=None):
        lines = s.split("\n")
        lh = ls or size * 1.28
        y0 = y - (len(lines) - 1) * lh / 2
        for i, l in enumerate(lines):
            self.add(f'<text x="{x}" y="{y0 + i*lh:.1f}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
                     f'fill="{fill}" text-anchor="{anchor}" dominant-baseline="middle" {style}>{l}</text>')

    def line(self, pts, stroke=INK2, sw=1.6, arrow=True, dash=None, curve=False):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = ' marker-end="url(#ah)"' if arrow else ""
        if curve:
            (x1, y1), (cx, cy), (x2, y2) = pts
            path = f"M{x1},{y1} Q{cx},{cy} {x2},{y2}"
        else:
            path = "M" + " L".join(f"{x},{y}" for x, y in pts)
        self.add(f'<path d="{path}" fill="none" stroke="{stroke}" stroke-width="{sw}"{d}{m} stroke-linecap="round" stroke-linejoin="round"/>')

    def svg(self):
        defs = f'''<defs>
<marker id="ah" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
<path d="M0,0 L10,5 L0,10 z" fill="{INK2}"/></marker>
<marker id="ahv" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
<path d="M0,0 L10,5 L0,10 z" fill="{VIO}"/></marker>
<filter id="sh" x="-10%" y="-10%" width="120%" height="130%"><feDropShadow dx="0" dy="1.5" stdDeviation="2" flood-color="#000" flood-opacity="0.10"/></filter>
<pattern id="wm" width="6" height="6" patternUnits="userSpaceOnUse"><rect width="6" height="6" fill="none"/>
<circle cx="1.5" cy="1.5" r="0.7" fill="{BLUE}" opacity="0.35"/><circle cx="4.5" cy="4.5" r="0.7" fill="{ORG}" opacity="0.3"/></pattern>
</defs>'''
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}">'
                f'<rect width="100%" height="100%" fill="#ffffff"/>{defs}' + "".join(self.e) + "</svg>")


def photo(s, x, y, w, h, marked=False, crop=None, rot=0):
    """Stylised photograph: sky, sun, two hills. Optional watermark texture, crop window, rotation."""
    cx, cy = x + w / 2, y + h / 2
    g = f'<g transform="rotate({rot} {cx} {cy})">' if rot else "<g>"
    s.add(g)
    s.add(f'<clipPath id="c{int(x)}{int(y)}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5"/></clipPath>')
    s.add(f'<g clip-path="url(#c{int(x)}{int(y)})">')
    s.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#dbe9f8"/>')
    s.add(f'<circle cx="{x+w*0.74}" cy="{y+h*0.3}" r="{h*0.12}" fill="#f6c453"/>')
    s.add(f'<path d="M{x},{y+h*0.78} Q{x+w*0.28},{y+h*0.42} {x+w*0.58},{y+h*0.76} T{x+w},{y+h*0.66} L{x+w},{y+h} L{x},{y+h} z" fill="#7cb87c"/>')
    s.add(f'<path d="M{x},{y+h*0.9} Q{x+w*0.45},{y+h*0.64} {x+w},{y+h*0.88} L{x+w},{y+h} L{x},{y+h} z" fill="#4f9a5a"/>')
    if marked:
        s.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="url(#wm)"/>')
    s.add("</g>")
    s.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" fill="none" stroke="{INK3}" stroke-width="1"/>')
    if crop:
        s.add(f'<rect x="{x+w*crop[0]}" y="{y+h*crop[1]}" width="{w*crop[2]}" height="{h*crop[3]}" fill="none" stroke="{RED}" stroke-width="1.6" stroke-dasharray="4 3"/>')
    s.add("</g>")


def bitstrip(s, x, y, n=32, cell=7, seed=3, colors=(BLUE, "#ffffff")):
    rnd = random.Random(seed)
    for i in range(n):
        b = rnd.random() > 0.5
        s.add(f'<rect x="{x+i*cell}" y="{y}" width="{cell-1}" height="{cell+3}" rx="1" fill="{colors[0] if b else colors[1]}" stroke="{colors[0]}" stroke-width="0.6"/>')


def key_icon(s, x, y, c=VIO, sc=1.0):
    s.add(f'<g transform="translate({x},{y}) scale({sc})"><circle cx="0" cy="0" r="6" fill="none" stroke="{c}" stroke-width="2.4"/>'
          f'<path d="M6,0 L20,0 M15,0 L15,5 M19,0 L19,4" stroke="{c}" stroke-width="2.4" stroke-linecap="round"/></g>')


def chip(s, x, y, w, h, title, sub, fill, stroke, tsize=12.5, ssize=10.5):
    s.add(f'<g filter="url(#sh)">')
    s.rect(x, y, w, h, fill=fill, stroke=stroke, sw=1.3, rx=9)
    s.add("</g>")
    if sub:
        s.text(x + w / 2, y + h / 2 - 8, title, size=tsize, weight=600)
        s.text(x + w / 2, y + h / 2 + 10, sub, size=ssize, fill=INK2)
    else:
        s.text(x + w / 2, y + h / 2, title, size=tsize, weight=600)


# =====================================================================================  FIG 1
def fig_framework():
    s = SVG(1600, 760)
    # lanes
    lanes = [(20, 600, "REGISTRAR", "holds key K", VIO_L), (640, 330, "PUBLIC CHANNEL", "attacks and adversaries", RED_L),
             (990, 590, "TRUSTED VERIFIER", "holds key K", GRN_L)]
    for x, w, t, sub, c in lanes:
        s.rect(x, 20, w, 600, fill=c, stroke="none", rx=18, op=0.55)
        s.text(x + 24, 52, t, size=14, weight=700, fill=INK, anchor="start", style='letter-spacing="1.5"')
        s.text(x + 24, 74, sub, size=12, fill=INK2, anchor="start")
    key_icon(s, 590, 50); key_icon(s, 1550, 50)

    # ---- registrar
    # payload card
    s.add('<g filter="url(#sh)">'); s.rect(45, 110, 170, 150, fill="#fff", stroke=LINE, rx=10); s.add("</g>")
    s.text(130, 132, "Payload P", size=13.5, weight=700)
    rows = [("author ID", "24 bit"), ("day index", "16 bit"), ("content hash", "8 bit")]
    for i, (a, b) in enumerate(rows):
        yy = 162 + i * 30
        s.rect(58, yy - 12, 144, 24, fill=GRY_L, stroke="none", rx=5)
        s.text(66, yy, a, size=11.5, anchor="start", fill=INK)
        s.text(195, yy, b, size=11, anchor="end", fill=INK2)
    s.text(130, 280, "48 bits", size=11.5, fill=INK2)

    # codec stack
    s.add('<g filter="url(#sh)">'); s.rect(250, 100, 190, 300, fill="#fff", stroke=VIO_M, sw=1.4, rx=12); s.add("</g>")
    s.text(345, 124, "Keyed codec", size=13.5, weight=700, fill=VIO)
    steps = [("1  Tag", "τ = HMAC(K₂, P), 23 bit"), ("2  Encrypt", "C = AES-CTR(K₁, P), IV = τ"),
             ("3  Error-correct", "BCH(127,71) on C‖τ"), ("4  Whiten + permute", "⊕ mask M(K₃), π(K₃)")]
    for i, (a, b) in enumerate(steps):
        yy = 150 + i * 60
        s.rect(264, yy, 162, 48, fill=VIO_L, stroke="none", rx=8)
        s.text(345, yy + 16, a, size=12, weight=600, fill=INK)
        s.text(345, yy + 34, b, size=10.3, fill=INK2)
        if i < 3:
            s.line([(345, yy + 49), (345, yy + 59)], sw=1.3)
    s.line([(216, 185), (248, 185)])
    # 128-bit word
    s.text(345, 425, "128-bit word w", size=12, weight=600)
    bitstrip(s, 233, 440, n=32, cell=7, seed=5, colors=(VIO, "#fff"))
    s.line([(345, 401), (345, 412)], sw=1.3)

    # host encoder
    s.line([(457, 446), (470, 446), (470, 330), (482, 330)])
    s.add('<g filter="url(#sh)">'); s.rect(484, 270, 136, 120, fill="#fff", stroke=BLUE_M, sw=1.4, rx=12); s.add("</g>")
    # mini U-Net glyph
    for i, (hh, xx) in enumerate([(44, 498), (32, 516), (20, 534), (32, 552), (44, 570)]):
        s.rect(xx, 324 - hh / 2, 12, hh, fill=BLUE_M if i != 2 else BLUE, stroke="none", rx=2)
    s.text(552, 290, "Host encoder", size=12.5, weight=700, fill=BLUE)
    s.text(552, 368, "ours, Pixel Seal\nor Video Seal", size=9.8, fill=INK2, ls=12)
    # cover image into encoder
    photo(s, 498, 160, 100, 72)
    s.text(548, 146, "cover image", size=11, fill=INK2)
    s.line([(548, 234), (548, 268)])
    # marked image
    photo(s, 498, 470, 100, 72, marked=True)
    s.text(548, 556, "marked image", size=11.5, weight=600)
    s.text(548, 574, "PSNR 39 dB (ours)", size=10, fill=INK2)
    s.line([(552, 391), (552, 466)])

    # ---- channel
    s.line([(600, 506), (650, 506), (650, 233), (681, 233)])
    photo(s, 685, 110, 92, 66, marked=True, crop=(0.2, 0.18, 0.6, 0.64))
    photo(s, 800, 110, 92, 66, marked=True, rot=-12)
    photo(s, 685, 200, 92, 66, marked=True)
    s.add(f'<rect x="685" y="200" width="92" height="66" fill="#999" opacity="0.18"/>')
    photo(s, 800, 200, 92, 66, marked=True)
    s.add(f'<rect x="800" y="200" width="92" height="66" fill="#fff" opacity="0.35"/>')
    s.text(731, 190, "crop", size=10.5, fill=INK2); s.text(846, 190, "rotate", size=10.5, fill=INK2)
    s.text(731, 280, "JPEG / WebP", size=10.5, fill=INK2); s.text(846, 280, "noise, blur, gamma", size=10.5, fill=INK2)
    s.text(788, 312, "27 attack conditions", size=12, weight=600)
    # adversaries
    s.add('<g filter="url(#sh)">'); s.rect(665, 350, 280, 160, fill="#fff", stroke="#f2b8b8", sw=1.3, rx=12); s.add("</g>")
    s.text(805, 372, "Adversaries (Sec. 3)", size=12.5, weight=700, fill=RED)
    adv = [("A0", "unmarked image presented"), ("A3", "forge a payload without K"), ("A4", "copy a mark to a new image"),
           ("A5", "collage of marked content"), ("A6", "overwrite with the same model")]
    for i, (a, b) in enumerate(adv):
        yy = 398 + i * 22
        s.rect(680, yy - 9, 30, 18, fill=RED_L, stroke="none", rx=4)
        s.text(695, yy, a, size=10.5, weight=700, fill=RED)
        s.text(720, yy, b, size=11, anchor="start")
    s.line([(805, 511), (805, 560), (1010, 560), (1010, 520)])
    s.text(905, 574, "image x′ presented for verification", size=11, fill=INK2)

    # ---- verifier
    s.add('<g filter="url(#sh)">'); s.rect(1012, 400, 150, 118, fill="#fff", stroke=BLUE_M, sw=1.4, rx=12); s.add("</g>")
    s.text(1087, 420, "Host decoder", size=12.5, weight=700, fill=BLUE)
    for i, hh in enumerate([50, 40, 30, 22, 16]):
        s.rect(1032 + i * 16, 462 - hh / 2, 11, hh, fill=BLUE_M, stroke="none", rx=2)
    s.text(1087, 500, "no key · outputs bits", size=10, fill=INK2)
    # hypotheses
    s.line([(1163, 459), (1192, 459)])
    s.add('<g filter="url(#sh)">'); s.rect(1194, 380, 168, 158, fill="#fff", stroke=LINE, sw=1.3, rx=12); s.add("</g>")
    s.text(1278, 400, "Sync hypotheses", size=12.5, weight=700)
    for j, n in enumerate([1, 2, 3]):
        bx, by, sz = 1210 + j * 50, 425, 40
        s.rect(bx, by, sz, sz, fill=GRY_L, stroke=INK3, sw=0.8, rx=3)
        for k in range(1, n):
            s.add(f'<line x1="{bx+sz*k/n}" y1="{by}" x2="{bx+sz*k/n}" y2="{by+sz}" stroke="{INK3}" stroke-width="0.8"/>')
            s.add(f'<line x1="{bx}" y1="{by+sz*k/n}" x2="{bx+sz}" y2="{by+sz*k/n}" stroke="{INK3}" stroke-width="0.8"/>')
        s.text(bx + sz / 2, by + sz + 12, f"{n}×{n}", size=10, fill=INK2)
    s.text(1278, 498, "h = 1 … N<tspan baseline-shift='sub' font-size='75%'>geo</tspan> (≤ 8)", size=10.5, fill=INK2)
    s.text(1278, 518, "global mode: N = 1", size=10.5, weight=600, fill=INK)
    # keyed check
    s.line([(1363, 459), (1392, 459)])
    s.add('<g filter="url(#sh)">'); s.rect(1394, 360, 182, 198, fill="#fff", stroke=VIO_M, sw=1.4, rx=12); s.add("</g>")
    s.text(1485, 380, "Keyed check", size=12.5, weight=700, fill=VIO)
    chk = ["un-permute, un-mask", "BCH decode, radius 9", "decrypt C with IV = τ", "recompute tag → equal?"]
    for i, c in enumerate(chk):
        yy = 410 + i * 30
        s.rect(1408, yy - 11, 154, 22, fill=VIO_L, stroke="none", rx=5)
        s.text(1485, yy, c, size=10.6)
    s.text(1485, 535, "first accepted h wins", size=10, fill=INK2)
    # outcomes
    s.line([(1485, 360), (1485, 316), (1440, 316), (1440, 300)])
    s.line([(1530, 360), (1530, 300)])
    s.add('<g filter="url(#sh)">'); s.rect(1365, 205, 135, 92, fill=GRN_L, stroke="#9fd19f", sw=1.3, rx=10); s.add("</g>")
    s.text(1432, 226, "✓ accept", size=13, weight=700, fill=GRN)
    s.text(1432, 252, "payload P:", size=10.8, fill=INK2)
    s.text(1432, 272, "author, day, hash", size=10.8)
    s.add('<g filter="url(#sh)">'); s.rect(1510, 205, 66, 92, fill=RED_L, stroke="#f2b8b8", sw=1.3, rx=10); s.add("</g>")
    s.text(1543, 240, "✗", size=18, weight=700, fill=RED)
    s.text(1543, 266, "reject", size=11.5, weight=600, fill=RED)
    # verifier text
    s.text(1015, 120, "Decision rule", size=13, weight=700, anchor="start")
    s.text(1015, 146, "accept only if some hypothesis passes", size=11.5, anchor="start", fill=INK2)
    s.text(1015, 166, "BCH decoding and the 23-bit tag check", size=11.5, anchor="start", fill=INK2)

    # ---- guarantee band
    s.add('<g filter="url(#sh)">'); s.rect(20, 640, 1560, 100, fill="#fff", stroke=LINE, sw=1.2, rx=14); s.add("</g>")
    key_icon(s, 52, 690, c=VIO, sc=1.1)
    s.text(88, 668, "Proven (Sec. 4)", size=12.5, weight=700, anchor="start", fill=VIO)
    s.text(88, 692, "False acceptance on any key-independent image ≤ N<tspan baseline-shift='sub' font-size='75%'>geo</tspan> · p<tspan baseline-shift='sub' font-size='75%'>BCH</tspan>(r) · 2⁻²³"
           "   →   3.2·10⁻¹¹ for G9 with N = 1", size=12, anchor="start")
    s.text(88, 716, "Forgery without K bounded per query  ·  payload confidentiality (SIV)", size=12, anchor="start", fill=INK2)
    s.add(f'<line x1="900" y1="660" x2="900" y2="722" stroke="{LINE}" stroke-width="1"/>')
    s.text(925, 668, "Measured (Sec. 6)", size=12.5, weight=700, anchor="start", fill=RED)
    s.text(925, 692, "0 / 250 000 false accepts on unmarked images  ·  90.6 % verified over 27 attacks", size=12, anchor="start")
    s.text(925, 716, "blind copy 0 %  ·  oracle copy 4.3 % after hash  ·  same-model overwrite erases the mark", size=12, anchor="start", fill=INK2)
    return s.svg()


# =====================================================================================  FIG 3
def cuboid(s, x, y, w, h, d, fill, edge, label=None, sub=None, lab_dy=16):
    """front face at (x, y) size w x h; depth d drawn up-right."""
    dx, dy = d * 0.6, -d * 0.45
    top = f"M{x},{y} L{x+dx},{y+dy} L{x+w+dx},{y+dy} L{x+w},{y} z"
    side = f"M{x+w},{y} L{x+w+dx},{y+dy} L{x+w+dx},{y+h+dy} L{x+w},{y+h} z"
    s.add(f'<path d="{side}" fill="{fill}" stroke="{edge}" stroke-width="0.9" opacity="0.75"/>')
    s.add(f'<path d="{top}" fill="{fill}" stroke="{edge}" stroke-width="0.9" opacity="0.55"/>')
    s.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{edge}" stroke-width="0.9"/>')
    if label:
        s.text(x + w / 2 + dx / 2, y + h + lab_dy, label, size=10.5, weight=600)
    if sub:
        s.text(x + w / 2 + dx / 2, y + h + lab_dy + 15, sub, size=9.6, fill=INK2)
    return x + w + dx, y + h / 2


def fig_network():
    s = SVG(1600, 900)
    # ---------------- (a) encoder
    s.text(30, 34, "(a)  Encoder: U-Net with a spatial message map", size=15, weight=700, anchor="start")
    s.text(30, 56, "2.03 M parameters · input resized to 256×256 · residual upsampled back to full resolution", size=11.5, fill=INK2, anchor="start")
    mid = 230
    photo(s, 30, mid - 55, 110, 110)
    s.text(85, mid + 72, "cover C", size=11, weight=600); s.text(85, mid + 87, "3×256×256", size=9.6, fill=INK2)
    # message
    s.text(85, 352, "message w", size=11, weight=600)
    bitstrip(s, 30, 362, n=16, cell=7, seed=11, colors=(VIO, "#fff"))
    s.text(85, 388, "128 bit", size=9.6, fill=INK2)
    # FC -> 16x8x8
    ex, ey = cuboid(s, 175, 345, 26, 26, 22, VIO_L, VIO, "FC → 16×8×8", None, lab_dy=26)
    s.line([(144, 368), (172, 358)])
    s.text(230, 410, "bilinear ↑ to 256²", size=9.6, fill=INK2)
    # encoder stages: (spatial label, channels, height px, depth px)
    stages = [("concat", "19×256²", 120, 10, GRY_L, INK3),
              ("DoubleConv", "64×256²", 120, 24, ORG_L, ORG),
              ("pool + DoubleConv", "128×128²", 84, 36, ORG_L, ORG),
              ("pool + DoubleConv", "256×64²", 58, 52, ORG_M, ORG),
              ("↑ + skip + DoubleConv", "128×128²", 84, 36, ORG_L, ORG),
              ("↑ + skip + DoubleConv", "64×256²", 120, 24, ORG_L, ORG),
              ("concat C, 1×1 conv, tanh", "residual R: 3×256²", 120, 8, BLUE_L, BLUE)]
    xs = [300, 405, 525, 650, 790, 920, 1060]
    pos = []
    for (lab, sub, h, d, fc, ec), x in zip(stages, xs):
        ox, oy = cuboid(s, x, mid - h / 2, 30 if d > 9 else 14, h, d, fc, ec, lab, sub, lab_dy=18 + (120 - h) / 2)
        pos.append((x, ox, mid))
    s.line([(141, mid), (296, mid)])
    s.line([(ex + 4, 352), (296, mid + 40)])
    for i in range(len(pos) - 1):
        s.line([(pos[i][1] + 4, mid), (pos[i + 1][0] - 4, mid)])
    # skip connections
    s.line([(pos[2][0] + 22, mid - 52), ((pos[2][0] + pos[4][0]) / 2 + 20, mid - 135), (pos[4][0] + 20, mid - 50)], curve=True, dash="5 4", stroke=ORG)
    s.line([(pos[1][0] + 20, mid - 70), ((pos[1][0] + pos[5][0]) / 2 + 15, mid - 205), (pos[5][0] + 18, mid - 68)], curve=True, dash="5 4", stroke=ORG)
    s.text((pos[1][0] + pos[5][0]) / 2 + 15, mid - 112, "skip connections", size=10.5, fill=ORG, weight=600)
    # residual add
    ax_ = 1180
    s.line([(pos[6][1] + 4, mid), (ax_ - 18, mid)])
    s.add(f'<circle cx="{ax_}" cy="{mid}" r="16" fill="#fff" stroke="{INK2}" stroke-width="1.4"/>')
    s.text(ax_, mid + 1, "+", size=20, weight=600)
    s.text(ax_ - 12, mid - 34, "S = C + β·R", size=11.5, weight=600, anchor="end")
    s.line([(85, mid - 58), (85, 88), (ax_, 88), (ax_, mid - 18)], stroke=INK3, sw=1.2)
    s.text(100, 78, "cover C at full resolution", size=10, fill=INK3, anchor="start")
    s.line([(ax_ + 17, mid), (1250, mid)])
    photo(s, 1255, mid - 55, 110, 110, marked=True)
    s.text(1310, mid + 72, "marked image S", size=11, weight=600)
    # training box
    s.add('<g filter="url(#sh)">'); s.rect(1395, 120, 185, 230, fill="#fff", stroke=LINE, rx=12); s.add("</g>")
    s.text(1487, 142, "Training only", size=12.5, weight=700)
    tr = [("noise layer", "27-type curriculum"), ("discriminator", "adversarial loss"), ("controller", "PSNR / bit-accuracy"), ("losses", "MSE, SSIM, BCE")]
    for i, (a, b) in enumerate(tr):
        yy = 175 + i * 42
        s.rect(1408, yy - 14, 160, 36, fill=GRY_L, stroke="none", rx=6)
        s.text(1488, yy - 2, a, size=11, weight=600); s.text(1488, yy + 13, b, size=9.8, fill=INK2)

    # divider
    s.add(f'<line x1="30" y1="470" x2="1570" y2="470" stroke="{LINE}" stroke-width="1"/>')

    # ---------------- (b) decoder
    s.text(30, 500, "(b)  Decoder with a spatial transformer", size=15, weight=700, anchor="start")
    s.text(30, 522, "0.87 M parameters · keyless · one decode per tile; tile logits averaged per hypothesis", size=11.5, fill=INK2, anchor="start")
    mid2 = 690
    photo(s, 30, mid2 - 55, 110, 110, marked=True, rot=-10)
    s.text(85, mid2 + 75, "attacked image x′", size=11, weight=600); s.text(85, mid2 + 90, "(or one tile) 3×256²", size=9.6, fill=INK2)
    # STN block
    s.add('<g filter="url(#sh)">'); s.rect(175, 560, 300, 250, fill="#fff", stroke=BLUE_M, sw=1.4, rx=12); s.add("</g>")
    s.text(325, 582, "Spatial transformer", size=12.5, weight=700, fill=BLUE)
    loc = [("conv 5×5, 16, s2", 48, 10), ("conv 5×5, 32, s2", 34, 14), ("conv 3×3, 32, s2", 24, 14)]
    lx = 195
    for i, (lab, h, d) in enumerate(loc):
        cuboid(s, lx + i * 52, 650 - h / 2, 16, h, d, BLUE_L, BLUE)
    s.text(270, 612, "localization net", size=10.5, weight=600, fill=INK2)
    s.text(270, 705, "conv 16 → 32 → 32, stride 2", size=9.6, fill=INK2)
    s.text(270, 720, "pool 4×4 → FC 64 → FC 6", size=9.6, fill=INK2)
    s.line([(345, 650), (372, 650)])
    s.rect(375, 628, 82, 44, fill=BLUE_L, stroke="none", rx=6)
    s.text(416, 643, "affine θ", size=11, weight=600); s.text(416, 659, "2×3", size=9.6, fill=INK2)
    # warp illustration
    s.add(f'<polygon points="215,770 268,752 284,792 230,808" fill="{GRY_L}" stroke="{INK3}" stroke-width="1"/>')
    s.line([(292, 780), (338, 780)], stroke=BLUE)
    s.rect(346, 756, 50, 46, fill=GRY_L, stroke=INK3, sw=1, rx=2)
    s.text(420, 780, "warp back", size=10, fill=INK2, anchor="start")
    s.line([(416, 674), (416, 745), (372, 745), (372, 754)], stroke=BLUE, sw=1.2)
    s.line([(141, mid2), (172, mid2)])
    # conv body
    body = [(256, 120, 18), (256, 120, 18), (128, 84, 18), (128, 84, 18), (64, 58, 18), (64, 58, 18), (32, 40, 18)]
    bx = 520
    prev = (476, mid2)
    s.line([prev, (516, mid2)])
    for i, (sp, h, d) in enumerate(body):
        x = bx + i * 62
        ox, oy = cuboid(s, x, mid2 - h / 2, 18, h, d, ORG_L if sp > 64 else ORG_M, ORG)
        if i:
            s.line([(x - 26, mid2), (x - 3, mid2)], sw=1.2)
    s.text(bx + 3 * 62 + 20, mid2 + 92, "7 × (conv 3×3, BN, ReLU), 64 channels", size=10.5, weight=600)
    s.text(bx + 3 * 62 + 20, mid2 + 108, "256² → 128² → 64² → 32²  (stride 2 in blocks 3, 5, 7)", size=9.6, fill=INK2)
    # pool + head
    px = bx + 7 * 62 + 10
    s.line([(px - 20, mid2), (px, mid2)])
    cuboid(s, px + 4, mid2 - 14, 14, 28, 18, ORG_M, ORG, "avg pool", "64×4×4", lab_dy=60)
    hx = px + 70
    s.line([(px + 34, mid2), (hx - 4, mid2)])
    for i, (n, h) in enumerate([(1024, 150), (512, 104), (128, 60)]):
        x = hx + i * 62
        s.rect(x, mid2 - h / 2, 16, h, fill=VIO_L if i == 2 else BLUE_L, stroke=VIO if i == 2 else BLUE, sw=1, rx=3)
        s.text(x + 8, mid2 + h / 2 + 16, str(n), size=10, weight=600)
        if i:
            s.line([(x - 44, mid2), (x - 3, mid2)], sw=1.2)
    s.text(hx + 70, mid2 - 92, "FC + ReLU + FC", size=10.5, weight=600)
    # output bits
    s.line([(hx + 142, mid2), (hx + 178, mid2)])
    bitstrip(s, hx + 184, mid2 - 6, n=16, cell=7, seed=11, colors=(VIO, "#fff"))
    s.text(hx + 240, mid2 + 26, "128 logits → bits y", size=10.5, weight=600)
    s.text(hx + 240, mid2 + 42, "to the keyed check (Fig. 2)", size=9.6, fill=INK2)
    # ablation callout
    s.add('<g filter="url(#sh)">'); s.rect(1240, 790, 330, 70, fill=BLUE_L, stroke="none", rx=10); s.add("</g>")
    s.text(1405, 812, "Ablation (Sec. 6.5)", size=11.5, weight=700, fill=BLUE)
    s.text(1405, 836, "STN bypassed at test time: rotation 15°", size=10.5)
    s.text(1405, 851, "verification 96.8 % → 0.8 %", size=10.5, weight=600)
    return s.svg()


if __name__ == "__main__":
    import sys
    out = sys.argv[1]
    open(out + "/fig1_framework.svg", "w").write(fig_framework())
    open(out + "/fig3_network.svg", "w").write(fig_network())
    print("svg ok")
