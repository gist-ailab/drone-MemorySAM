"""논문 Fig. 1을 개별 편집 가능한 PowerPoint 도형으로 다시 그린다.

원본 TikZ: _paper/ieee-access-2026/latex/figures/fig_arch.tex
TikZ 좌표(cm, 위쪽이 +y)를 슬라이드 좌표(cm) X=x+0.15, Y=7.05-y로 변환한다.
실행: python3 _paper/ieee-access-2026/figures/src/make_fig_arch_pptx.py
"""

from pathlib import Path

from PIL import ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Cm, Pt


BASE = Path(__file__).resolve().parents[2]
FIGURES = BASE / "latex" / "figures"
THUMBS = FIGURES / "fig1_thumbs"
OUTPUT = FIGURES / "fig_arch_editable.pptx"

C = {
    "rgb": "6929C4", "depth": "00539A", "event": "8E6A00",
    "lidar": "9F1853", "acc": "007D79", "ink": "3C3838",
    "mut": "565151", "bd": "CAC5C4", "sf": "F7F3F2",
    "g20": "E5E0DF", "white": "FFFFFF",
}

FONT_REGULAR = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"


def blend(hex_color, percent):
    """TikZ의 color!percent (나머지는 흰색)와 같은 색을 반환한다."""
    ratio = percent / 100
    return "".join(f"{round(int(hex_color[i:i+2], 16)*ratio + 255*(1-ratio)):02X}"
                   for i in (0, 2, 4))


def color(key):
    return RGBColor.from_string(C.get(key, key))


def xy(x, y):
    return Cm(x + 0.15), Cm(7.05 - y)


def set_dash(shape, dash="sysDash"):
    ln = shape.get_or_add_ln()
    prst = OxmlElement("a:prstDash")
    prst.set("val", dash)
    ln.append(prst)


def arrow(shape, begin=False, end=True):
    ln = shape.get_or_add_ln()
    for tag in (("headEnd", begin), ("tailEnd", end)):
        if tag[1]:
            node = OxmlElement("a:" + tag[0])
            node.set("type", "triangle")
            node.set("w", "sm")
            node.set("len", "sm")
            ln.append(node)


def rect(slide, name, x1, y1, x2, y2, *, stroke="mut", fill="white",
         width=0.45, rounded=False, dashed=False, radius_pt=1.5):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    shp = slide.shapes.add_shape(kind, Cm(x1 + 0.15), Cm(7.05-y2),
                                 Cm(x2-x1), Cm(y2-y1))
    shp.name = name
    shp.fill.solid()
    shp.fill.fore_color.rgb = color(fill)
    shp.line.color.rgb = color(stroke)
    shp.line.width = Pt(width)
    if dashed:
        set_dash(shp)
    if rounded:
        # PowerPoint 의 모서리 조정값은 짧은 변에 대한 비율이다. TikZ 의 rounded corners
        # 반지름(pt)을 상자 크기에 맞춰 환산해야 큰 상자도 원본처럼 살짝만 둥글다.
        shp.adjustments[0] = min(0.5, (radius_pt * 2.54 / 72) / min(x2 - x1, y2 - y1))
    return shp


def circle(slide, name, x, y, diameter, fill, stroke=None, width=0.45):
    shp = slide.shapes.add_shape(MSO_SHAPE.OVAL, Cm(x+0.15-diameter/2),
                                 Cm(7.05-y-diameter/2), Cm(diameter), Cm(diameter))
    shp.name = name
    shp.fill.solid()
    shp.fill.fore_color.rgb = color(fill)
    if stroke:
        shp.line.color.rgb = color(stroke)
        shp.line.width = Pt(width)
    else:
        shp.line.fill.background()
    return shp


def line(slide, name, start, finish, *, stroke="mut", width=0.45,
         head=False, tail=False, dashed=False):
    shp = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
                                     *xy(*start), *xy(*finish))
    shp.name = name
    shp.line.color.rgb = color(stroke)
    shp.line.width = Pt(width)
    if dashed:
        set_dash(shp, "dash")
    if head or tail:
        arrow(shp, begin=head, end=tail)
    return shp


def path(slide, name, points, *, stroke="mut", width=0.45,
         head=False, tail=True, dashed=False):
    """꺾은선의 마지막 세그먼트에만 화살표를 둔다."""
    for idx, (a, b) in enumerate(zip(points, points[1:])):
        line(slide, f"{name} {idx+1}", a, b, stroke=stroke, width=width,
             head=head and idx == 0, tail=tail and idx == len(points)-2,
             dashed=dashed)


def text_size(lines, pt, bold=False):
    font_file = FONT_BOLD if bold else FONT_REGULAR
    font = ImageFont.truetype(font_file, round(pt*8))
    widest = max(font.getlength(s) for s in lines) / 8
    # A small glyph overhang allowance keeps PowerPoint from clipping letters.
    return max(0.13, widest * 2.54/72 + 0.035), len(lines) * (pt+1.0) * 2.54/72


def label(slide, name, x, y, value, *, anchor="center", size=6.5,
          bold=False, ink="ink", italic=False, rotate=0, widths=None):
    lines = value.split("\\")
    w, h = text_size(lines, size, bold)
    if widths is not None:
        w = widths
    if rotate:
        # After the 270-degree rotation, the visual right edge meets the
        # image's west edge; the visual centre remains at the lane's y.
        left = x - h/2 - w/2
        top = y + h/2
        align = PP_ALIGN.CENTER
        vertical = MSO_ANCHOR.MIDDLE
    else:
        left = x - (w if "east" in anchor else w/2 if "west" not in anchor else 0)
        top = y + (0 if "north" in anchor else h if "south" in anchor else h/2)
        align = PP_ALIGN.RIGHT if "east" in anchor else PP_ALIGN.LEFT if "west" in anchor else PP_ALIGN.CENTER
        vertical = MSO_ANCHOR.TOP if "north" in anchor else MSO_ANCHOR.BOTTOM if "south" in anchor else MSO_ANCHOR.MIDDLE
    shp = slide.shapes.add_textbox(Cm(left+0.15), Cm(7.05-top), Cm(w), Cm(h))
    shp.name = name
    shp.rotation = rotate
    tf = shp.text_frame
    tf.clear()
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.word_wrap = False
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.vertical_anchor = vertical
    for i, content in enumerate(lines):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.space_before = Pt(0)
        para.space_after = Pt(0)
        run = para.add_run()
        run.text = content
        run.font.name = "Arial"
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.italic = italic
        run.font.color.rgb = color(ink)
    return shp


def loss(slide, name, x, y, suffix, ink="ink"):
    shp = label(slide, name, x, y, "L" + suffix, anchor="west", ink=ink)
    para = shp.text_frame.paragraphs[0]
    para.clear()
    first = para.add_run()
    first.text = "L"
    first.font.name = "Arial"
    first.font.size = Pt(8)
    first.font.italic = True
    first.font.color.rgb = color(ink)
    sub = para.add_run()
    sub.text = suffix
    sub.font.name = "Arial"
    sub.font.size = Pt(6)
    sub.font.color.rgb = color(ink)
    sub._r.get_or_add_rPr().set("baseline", "-25000")
    return shp


def thumbnail(slide, name, filename, x, y, width, border=None, border_pt=0.9):
    img = slide.shapes.add_picture(str(THUMBS / filename), Cm(x+0.15-width/2),
                                   Cm(7.05-y-width/2), width=Cm(width))
    img.name = name
    if border:
        frame = rect(slide, name + " border", x-width/2, y-width/2,
                     x+width/2, y+width/2, stroke=border, width=border_pt)
        frame.fill.background()
    return img


def plus(slide, name, x, y, stroke="mut", width=0.45):
    circle(slide, name + " circle", x, y, 0.26, "white", stroke, width)
    line(slide, name + " vertical", (x, y+0.13), (x, y-0.13), stroke=stroke, width=width)
    line(slide, name + " horizontal", (x-0.13, y), (x+0.13, y), stroke=stroke, width=width)


def build():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Cm(18.4), Cm(7.3)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    lanes = [("RGB", "rgb", "rgb", 5.15), ("Depth", "depth", "depth", 3.9),
             ("Event", "event", "event", 2.65), ("LiDAR", "lidar", "lidar", 1.4)]

    label(slide, "Panel a", 0, 6.75, "(a)", anchor="west", size=8, bold=True)
    label(slide, "Inference title", 0.5, 6.75,
          "Inference: one frozen backbone, sensor-specific adapters, cross-modal fusion",
          anchor="west", size=6.8, bold=True)
    for sensor, filename, hue, yy in lanes:
        thumbnail(slide, sensor + " input", filename + ".png", 0.9, yy, 1.05, hue)
        label(slide, sensor + " name", 0.375, yy, sensor, size=6, bold=True,
              ink=hue, rotate=270)

    rect(slide, "Frozen DINOv3-L backbone", 1.8, 0.85, 5.0, 6.15,
         stroke="mut", fill="sf", rounded=True, dashed=True)
    label(slide, "DINOv3-L title", 1.92, 6.02, "DINOv3-L", anchor="north west", size=6.8, bold=True)
    label(slide, "Frozen shared label", 4.88, 6.0, "frozen, shared", anchor="north east", size=5.6, ink="mut")
    label(slide, "Tap label", 2.62, 5.52, "tap", anchor="east", size=5.6, ink="mut")
    for tap, tx in zip((6, 12, 18, 24), (3.25, 3.77, 4.29, 4.81)):
        label(slide, f"Tap {tap} heading", tx, 5.52, str(tap), size=5.6, ink="mut")
    for sensor, _, hue, yy in lanes:
        rect(slide, sensor + " LoRA", 1.97, yy-0.19, 2.62, yy+0.19,
             stroke=hue, fill=blend(C[hue], 14), width=0.6, rounded=True, radius_pt=0.8)
        label(slide, sensor + " LoRA text", 2.295, yy, "LoRA", size=5.2, bold=True, ink=hue)
        for k in range(4):
            x0 = 2.82 + 0.52*k
            rect(slide, f"{sensor} tap box {k+1}", x0, yy-0.16, x0+0.36, yy+0.16,
                 stroke="bd", fill="g20", width=0.35)
            circle(slide, f"{sensor} tap dot {k+1}", 3.25+0.52*k, yy, 0.09, hue)
        line(slide, sensor + " input flow", (1.425, yy), (1.97, yy), tail=True)

    label(slide, "Tokens heading", 5.5, 6.0, "tokens", size=5.6, ink="mut")
    for sensor, _, hue, yy in lanes:
        for k in (3, 2, 1, 0):
            x0, y0 = 5.25+0.06*k, yy-0.17+0.06*k
            rect(slide, f"{sensor} token stack {k+1}", x0, y0, x0+0.3, y0+0.3,
                 stroke=hue, fill=blend(C[hue], 20), width=0.4)

    rect(slide, "Cross-modal attention box", 6.2, 1.3, 8.4, 5.65,
         stroke="acc", width=0.7, rounded=True)
    label(slide, "Cross-modal attention title", 7.3, 5.12,
          "Cross-modal\\attention", size=6.8, bold=True)
    import math
    nodes = {}
    for sensor, angle in (("RGB", 135), ("Depth", 45), ("Event", -45), ("LiDAR", -135)):
        a = math.radians(angle)
        nodes[sensor] = (7.3+0.68*math.cos(a), 3.55+0.68*math.sin(a))
    pairs = (("RGB", "Depth"), ("RGB", "Event"), ("RGB", "LiDAR"),
             ("Depth", "Event"), ("Depth", "LiDAR"), ("Event", "LiDAR"))
    for a, b in pairs:
        x1, y1 = nodes[a]
        x2, y2 = nodes[b]
        dx, dy = x2-x1, y2-y1
        length = math.hypot(dx, dy)
        # TikZ shorten <=/>=1.5 mm, measured from the node centres.
        start = (x1+dx*0.15/length, y1+dy*0.15/length)
        end = (x2-dx*0.15/length, y2-dy*0.15/length)
        line(slide, f"{a} {b} bidirectional attention", start, end,
             stroke=blend(C["mut"], 70), width=0.4, head=True, tail=True)
    for sensor, _, hue, _ in lanes:
        circle(slide, sensor + " attention node", *nodes[sensor], 0.2, hue)
    label(slide, "Attention explanation", 7.3, 1.9,
          "each sensor attends\\to the other sensors\\2 layers", size=5.6, ink="mut")
    for sensor, yy in (("RGB", 4.85), ("Depth", 3.95), ("Event", 3.05), ("LiDAR", 2.2)):
        source_y = next(y for s, _, _, y in lanes if s == sensor)+0.05
        path(slide, sensor + " token to attention", [(5.8, source_y), (6.0, source_y),
             (6.0, yy), (6.2, yy)])

    rect(slide, "SimpleFPN outer box", 8.68, 1.7, 10.15, 5.65,
         stroke="bd", rounded=True)
    label(slide, "SimpleFPN title", 9.45, 5.38, "SimpleFPN", size=6.8, bold=True)
    for lab, w, yy in (("1/32", 0.32, 4.75), ("1/16", 0.44, 4.05),
                        ("1/8", 0.58, 3.2), ("1/4", 0.76, 2.2)):
        rect(slide, f"SimpleFPN {lab} feature", 9.6-w/2, yy-w/2, 9.6+w/2, yy+w/2)
        label(slide, f"SimpleFPN {lab} scale", 9.6-w/2-0.06, yy, lab,
              anchor="east", size=5.6, ink="mut")
    line(slide, "Attention to SimpleFPN", (8.4, 3.2), (8.68, 3.2), tail=True)
    plus(slide, "FPN sum", 9.6, 1.05)
    line(slide, "Sum to SimpleFPN", (9.6, 1.18), (9.6, 1.82), tail=True)
    label(slide, "Sum per level", 9.78, 1.05, "sum per level", anchor="west", size=5.6, ink="mut")

    path(slide, "LiDAR full resolution input", [(0.9, 0.875), (0.9, 0.35), (2.05, 0.35)])
    label(slide, "Full resolution input", 1.47, 0.3, "full-res. input",
          anchor="north", size=5.6, ink="mut")
    rect(slide, "Conv stem box", 2.05, 0.12, 3.45, 0.58, rounded=True, radius_pt=1.2)
    label(slide, "Conv stem text", 2.75, 0.35, "conv stem")
    line(slide, "Conv stem to extractors", (3.45, 0.35), (3.8, 0.35), tail=True)
    rect(slide, "Deformable extractors box", 3.8, 0.12, 6.75, 0.58, rounded=True, radius_pt=1.2)
    label(slide, "Deformable extractors text", 5.275, 0.35,
          "deformable extractors (×6)")
    line(slide, "Tap sampling", (4.35, 0.85), (4.35, 0.58), tail=True, dashed=True)
    label(slide, "Samples tap tokens", 4.42, 0.72, "samples tap tokens",
          anchor="west", size=5.6, ink="mut")
    path(slide, "High resolution path to FPN", [(6.75, 0.35), (9.6, 0.35), (9.6, 0.92)])
    label(slide, "High resolution path", 8.1, 0.38, "high-resolution path",
          anchor="south", size=5.6, ink="mut")

    path(slide, "FPN to dense head", [(10.15, 3.2), (10.3, 3.2),
         (10.3, 4.75), (10.5, 4.75)])
    line(slide, "FPN to query head", (10.3, 3.35), (10.5, 3.35), tail=True)
    rect(slide, "Dense head box", 10.5, 4.4, 11.95, 5.1,
         stroke="acc", width=0.7, rounded=True, radius_pt=1.2)
    label(slide, "Dense head title", 11.225, 4.87, "Dense head", size=6.8, bold=True)
    label(slide, "Per pixel label", 11.225, 4.6, "per-pixel", size=5.6, ink="mut")
    rect(slide, "Query head box", 10.5, 2.75, 11.95, 3.95,
         stroke="acc", width=0.7, rounded=True, radius_pt=1.2)
    label(slide, "Query head title", 11.225, 3.7, "Query head", size=6.8, bold=True)
    for k in range(8):
        circle(slide, f"Query dot {k+1}", 10.805+0.12*k, 3.36, 0.07,
               blend(C["acc"], 55))
    label(slide, "100 queries", 11.225, 3.03, "100 queries", size=5.6, ink="mut")
    plus(slide, "Arbiter", 12.35, 3.35, "acc", 0.6)
    path(slide, "Dense head to arbiter", [(11.95, 4.75), (12.75, 4.75),
         (12.75, 3.35), (12.48, 3.35)], stroke="acc", width=0.6)
    line(slide, "Query head to arbiter", (11.95, 3.35), (12.22, 3.35),
         stroke="acc", width=0.6, tail=True)
    label(slide, "Arbiter label", 12.45, 3.2, "arbiter", anchor="north west", size=5.6, ink="mut")
    thumbnail(slide, "Prediction thumbnail", "pred.png", 12.35, 1.45, 1.05, "ink", 0.5)
    line(slide, "Arbiter to prediction", (12.35, 3.22), (12.35, 1.975),
         stroke="acc", width=0.6, tail=True)
    label(slide, "Prediction label", 12.35, 0.925, "prediction",
          anchor="north", size=5.6, ink="mut")

    line(slide, "Training separator", (13.2, 0.05), (13.2, 6.6),
         stroke="bd", width=0.4)
    label(slide, "Panel b", 13.35, 6.75, "(b)", anchor="west", size=8, bold=True)
    label(slide, "Training title", 13.85, 6.75,
          "Training: clean and degraded passes", anchor="west", size=6.8, bold=True)
    thumbnail(slide, "Clean RGB thumbnail", "rgb.png", 14.1, 4.9, 0.95)
    label(slide, "Clean label", 14.1, 4.425, "clean", anchor="north", size=5.6, ink="mut")
    rect(slide, "Dropped sensor background", 13.775, 2.175, 14.725, 3.125,
         stroke="mut", width=0.4)
    line(slide, "Dropped sensor lower edge", (13.775, 2.175), (14.725, 2.175),
         stroke="mut", width=0.35)
    thumbnail(slide, "Degraded RGB thumbnail", "rgb_degraded.png", 14.1, 2.8, 0.95)
    label(slide, "Degraded input label", 14.2, 2.12,
          "degraded\\(blur, exposure,\\sensor dropout)",
          anchor="north", size=5.6, ink="mut")
    rect(slide, "Training model box", 14.95, 2.3, 16.45, 5.4,
         stroke="acc", width=0.7, rounded=True)
    label(slide, "Training model title", 15.7, 4.05, "Model (a)", size=6.8, bold=True)
    label(slide, "Same weights", 15.7, 3.62, "same\\weights", size=5.6, ink="mut")
    line(slide, "Clean to model", (14.575, 4.9), (14.95, 4.9), tail=True)
    line(slide, "Degraded to model", (14.725, 2.8), (14.95, 2.8), tail=True)
    rect(slide, "Frozen teacher box", 14.95, 0.3, 16.45, 1.5,
         stroke="mut", fill="sf", rounded=True, dashed=True)
    label(slide, "Teacher title", 15.7, 1.2, "Teacher", size=6.8, bold=True)
    label(slide, "Frozen clean trained", 15.7, 0.77,
          "frozen,\\clean-trained", size=5.6, ink="mut")
    path(slide, "Clean input to teacher", [(13.625, 4.9), (13.38, 4.9),
         (13.38, 0.9), (14.95, 0.9)])
    loss(slide, "Clean CE loss", 16.85, 4.9, "CE")
    loss(slide, "Degraded CE loss", 16.85, 3.2, "CE")
    loss(slide, "KD loss", 16.85, 2.4, "KD", "acc")
    line(slide, "Model clean CE flow", (16.45, 4.9), (16.85, 4.9), tail=True)
    path(slide, "Model degraded CE flow", [(16.45, 2.8), (16.65, 2.8),
         (16.65, 3.2), (16.85, 3.2)])
    path(slide, "Model KD flow", [(16.65, 2.8), (16.65, 2.4), (16.85, 2.4)])
    path(slide, "Teacher KD flow", [(16.45, 0.9), (16.95, 0.9),
         (16.95, 2.28)], stroke="acc", width=0.55, dashed=True)
    label(slide, "Soft targets", 17.25, 1.3, "soft\\targets\\T=2",
          anchor="west", size=5.6, ink="mut")
    # Math in the last line is italic in the TikZ original.
    soft = slide.shapes[-1].text_frame.paragraphs[-1].runs[0]
    soft.font.italic = True

    prs.save(OUTPUT)


if __name__ == "__main__":
    build()
