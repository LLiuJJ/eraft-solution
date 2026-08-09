"""生成简略架构图 PNG"""
from PIL import Image, ImageDraw, ImageFont

W, H = 900, 1200
img = Image.new("RGB", (W, H), "#ffffff")
draw = ImageDraw.Draw(img)

# 字体
try:
    font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    font_name = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    font_desc = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    font_shape = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 11)
except Exception:
    font_title = ImageFont.load_default()
    font_name = ImageFont.load_default()
    font_desc = ImageFont.load_default()
    font_shape = ImageFont.load_default()

# 颜色
C_TEXT = "#24292f"
C_DIM = "#57606a"
C_DINO = "#0969da"
C_INP = "#1a7f37"
C_FLOW = "#bf8700"
C_INOUT = "#8c959f"
C_SCORE = "#8250df"


def draw_block(x, y, w, h, border, fill, name, desc, shape):
    draw.rounded_rectangle([x, y, x + w, y + h], radius=8, outline=border, width=2, fill=fill)
    draw.text((x + w // 2, y + 8), name, fill=C_TEXT, font=font_name, anchor="mt")
    desc_lines = desc.split("\n") if desc else []
    for i, line in enumerate(desc_lines):
        draw.text((x + w // 2, y + 30 + i * 16), line, fill=C_DIM, font=font_desc, anchor="mt")
    if shape:
        base_y = y + 30 + len(desc_lines) * 16 + 4
        for i, line in enumerate(shape.split("\n")):
            draw.text((x + w // 2, base_y + i * 14), line, fill=C_DIM, font=font_shape, anchor="mt")
    return y + h


def draw_arrow(x, y, length=28):
    draw.line([(x, y), (x, y + length - 2)], fill="#8c959f", width=2)
    draw.polygon([(x - 5, y + length - 2), (x + 5, y + length - 2), (x, y + length + 4)], fill="#8c959f")
    return y + length + 8


def draw_dashed_rect(x, y, w, h, color, dash=8, gap=6):
    xx = x
    while xx < x + w:
        x2 = min(xx + dash, x + w)
        draw.line([(xx, y), (x2, y)], fill=color, width=2)
        xx += dash + gap
    xx = x
    while xx < x + w:
        x2 = min(xx + dash, x + w)
        draw.line([(xx, y + h), (x2, y + h)], fill=color, width=2)
        xx += dash + gap
    yy = y
    while yy < y + h:
        y2 = min(yy + dash, y + h)
        draw.line([(x, yy), (x, y2)], fill=color, width=2)
        yy += dash + gap
    yy = y
    while yy < y + h:
        y2 = min(yy + dash, y + h)
        draw.line([(x + w, yy), (x + w, y2)], fill=color, width=2)
        yy += dash + gap


# ── 绘制 ──
cx = W // 2
bw = 520
bx = cx - bw // 2

draw.text((W // 2, 20), "DINOv2 + INP-Former 架构图", fill=C_TEXT, font=font_title, anchor="mt")

y = 60

# 1. 输入
y = draw_block(bx, y, bw, 70, C_INOUT, "#f6f8fa",
    "输入：多视角图像", "5 个视角 × RGB × 518×518", "[B, 5, 3, 518, 518]") + 4
y = draw_arrow(cx, y) + 4

# 2. DINOv2
y = draw_block(bx, y, bw, 90, C_DINO, "#ddf4ff",
    "DINOv2 ViT-B/14  [冻结]",
    "自监督预训练特征提取 · 12 层 Transformer",
    "→ cls_tokens [B, 5, 768]\n→ patch_features [B, 5, 1369, 768]\n→ multi_scale_features (4 层)") + 4
y = draw_arrow(cx, y) + 4

# 3. INP-Former region
ry = y
rh = 300
rw = 560
rx = cx - rw // 2
draw_dashed_rect(rx, ry, rw, rh, C_INP)
# title pill
tw = 330
draw.rounded_rectangle([cx - tw // 2, ry - 14, cx + tw // 2, ry + 14], radius=14, outline=C_INP, width=2, fill="#ffffff")
draw.text((cx, ry - 1), "INP-Former (可逆神经过程 Transformer)", fill=C_INP, font=font_name, anchor="mm")

# ViewPatchEncoder
y = ry + 30
y = draw_block(bx, y, bw, 95, C_INP, "#dafbe1",
    "ViewPatchEncoder  [可训练]",
    "多视角 Transformer · 4 层 · 8 头注意力\n投影 1536→256 + 位置编码 + 视角嵌入",
    "→ cls_out [B, 256]\n→ view_tokens [B, 5, 256]\n→ patch_map [B, 5, 1369, 256]") + 4
y = draw_arrow(cx, y) + 4

# LayerNorm
lw = 300
lx = cx - lw // 2
y = draw_block(lx, y, lw, 40, "#d0d7de", "#f6f8fa",
    "LayerNorm", "稳定 Flow 输入分布", None) + 4
y = draw_arrow(cx, y) + 4

# Dual Flow
fw = 250
gap = 16
fx1 = cx - fw - gap // 2
fx2 = cx + gap // 2
fh = 85
fy = y

for fx, name, shape in [(fx1, "Flow_cls  [可训练]", "z_cls [B, 256]"),
                          (fx2, "Flow_view  [可训练]", "z_view [B, 5, 256]")]:
    draw.rounded_rectangle([fx, fy, fx + fw, fy + fh], radius=8, outline=C_FLOW, width=2, fill="#fff8c5")
    draw.text((fx + fw // 2, fy + 8), name, fill=C_TEXT, font=font_name, anchor="mt")
    draw.text((fx + fw // 2, fy + 30), "Normalizing Flow ×8 层", fill=C_DIM, font=font_desc, anchor="mt")
    draw.text((fx + fw // 2, fy + 46), "ActNorm → 旋转 → Coupling", fill=C_DIM, font=font_desc, anchor="mt")
    draw.text((fx + fw // 2, fy + 68), shape, fill=C_DIM, font=font_shape, anchor="mt")

y = fy + fh + 8
y = max(y, ry + rh) + 4
y = draw_arrow(cx, y) + 4

# 4. 输出
y = draw_block(bx, y, bw, 75, C_INOUT, "#f6f8fa",
    "输出", "z 空间 L2 距离 = 异常得分",
    "image_score = ‖z_cls‖ + mean(‖z_view‖)\npatch_score = k-NN + Flow 融合") + 4
y = draw_arrow(cx, y) + 4

# 5. 比赛提交
y = draw_block(bx, y, bw, 75, C_SCORE, "#fbefff",
    "比赛提交", "submission.csv + predicted_masks/",
    "图像级 anomaly_score\n像素级 448×448 mask") + 4

# Legend
ly = y + 25
legend_items = [
    ("DINOv2（冻结）", C_DINO, "#ddf4ff"),
    ("INP-Former（可训练）", C_INP, "#dafbe1"),
    ("Normalizing Flow", C_FLOW, "#fff8c5"),
    ("输入 / 输出", C_INOUT, "#f6f8fa"),
]
lx = cx - 320
for label, border, fill in legend_items:
    draw.rounded_rectangle([lx, ly, lx + 14, ly + 14], radius=3, outline=border, width=2, fill=fill)
    draw.text((lx + 20, ly + 1), label, fill=C_DIM, font=font_desc, anchor="lm")
    lx += 165

final = img.crop((0, 0, W, ly + 30))
final.save("img/architecture.png", "PNG")
print(f"Saved: img/architecture.png  size={final.size}")
