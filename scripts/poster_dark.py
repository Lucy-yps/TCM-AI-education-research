# -*- coding: utf-8 -*-
"""本草辨证 展板海报 0.6 x 1.5 m（竖向长幅，铺满版面）"""
import sys
from PIL import Image, ImageDraw, ImageFont, ImageFilter

MM_W, MM_H = 600, 1500
SCALE = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
OUT = sys.argv[2] if len(sys.argv) > 2 else "poster.png"
W_PX, H_PX = int(MM_W*SCALE), int(MM_H*SCALE)

BG=(40,63,65); BG2=(54,80,82); CARD=(60,88,90); CARD2=(72,102,104); LINE=(90,120,120)
ACID=(198,232,90); GOLD=(226,178,90); WHITE=(244,248,245)
GREY=(204,220,216); DIM=(160,182,178); RED=(240,110,96); BLUE=(120,200,228)

def px(mm): return int(mm*SCALE)
img = Image.new("RGB", (W_PX, H_PX), BG)
d = ImageDraw.Draw(img)
SHOT = "home_shot.png"
FD = "C:\\Windows\\Fonts\\"
def F(mm, bold=False, serif=False):
    p = FD + ("STZHONGS.TTF" if serif else ("msyhbd.ttc" if bold else "msyh.ttc"))
    try: return ImageFont.truetype(p, max(int(mm*SCALE), 6))
    except Exception: return ImageFont.load_default()
def T(x,y,s,mm,color=WHITE,bold=False,serif=False,anchor="la"):
    d.text((px(x),px(y)), s, font=F(mm,bold,serif), fill=color, anchor=anchor)
def R(x,y,w,h,c,rad=0,line=None,lw=0.3):
    d.rounded_rectangle([px(x),px(y),px(x+w),px(y+h)], radius=px(rad), fill=c,
                        outline=line, width=max(int(lw*SCALE),1) if line else 0)
def HL(x,y,w,c,t=0.4):
    d.rectangle([px(x),px(y),px(x+w),px(y+t)], fill=c)

# 背景：纵向渐变
for y in range(H_PX):
    t = y/H_PX
    k = 1 - abs(t-0.5)*1.4
    c = tuple(int(BG[i] + (BG2[i]-BG[i])*max(k,0)) for i in range(3))
    d.line([(0,y),(W_PX,y)], fill=c)
# 顶部装饰线
d.rectangle([0,0,W_PX,px(4)], fill=ACID)
d.rectangle([0,px(1496),W_PX,H_PX], fill=ACID)

X0, WD = 36, 528   # 左边距、内容宽
# ============ 标题区 (0-118) ============
T(X0, 26, "AIC 2026 · 算法创新赛 · AI+学科交叉", 8, ACID, False)
T(X0, 40, "本草辨证", 46, WHITE, True, True)
T(X0, 92, "面向中医教学与科研的知识检索与辨证训练工具", 11, GREY)
R(392, 26, 172, 84, CARD, 5)
T(478, 34, "92.9%", 26, ACID, True, anchor="ma")
T(478, 66, "病历体 Top-1", 7.5, GREY, anchor="ma")
T(478, 80, "TCM-SD 去泄漏 683 例", 6.5, DIM, anchor="ma")
HL(X0, 118, WD, DIM, 0.4)

# ============ 01 研究背景 (148-348) ============
y = 126
T(X0, y, "01", 16, ACID, True, True)
T(X0+22, y+3, "研究背景与痛点", 13, WHITE, True)
y += 26
bg = [
  ("术语鸿沟", "患者说「没胃口」，教材写「食少纳呆」。口语与规范术语之间缺少稳定的映射关系，学生只能靠记忆硬背。"),
  ("证据不可追溯", "通用大模型能给出证型名，却说不清依据了哪些症状。学生无法从结果反推辨证思路。"),
  ("不确定性缺失", "多数 AI 工具不区分「有把握」与「在猜」，也不告诉使用者这个结论有多可靠。"),
  ("分布失配被忽视", "口语体与病历体的文本分布差异极大，同一方法的准确率可能从 91.9% 跌到 12.7%。"),
  ("练习题稀缺", "编写一道带解析的病案题通常需要 30~60 分钟，难以规模化供给。"),
]
for i, (t, desc) in enumerate(bg):
    R(X0, y, WD, 32, CARD, 3)
    d.rectangle([px(X0), px(y), px(X0+3), px(y+32)], fill=GOLD)
    T(X0+12, y+5, ("①②③④⑤"[i]) + "  " + t, 9, WHITE, True)
    T(X0+12, y+18, desc, 6.8, GREY)
    y += 36

# ============ 02 解决方案 (320-620) ============
y = 342
T(X0, y, "02", 16, ACID, True, True)
T(X0+22, y+3, "解决方案：四层架构", 13, WHITE, True)
y += 26
arch = [
  ("应用层", "辨证训练 · 相似鉴别 · 方剂治法 · 舌脉辨识 · 学习报告", "六大功能模块"),
  ("决策层", "输入风格路由 · 置信度校准 · 正确性探针 · 保形预测 · 拒答", "回答「这个结果可靠吗」"),
  ("算法层", "TF-IDF / BM25 / RRF 混合检索 · 轻量分类器（7.6 MB）", "回答「有哪些候选」"),
  ("数据层", "知识库 37 证型 / 248 症状词 · TCM-SD 54,152 条真实病历", "回答「依据是什么」"),
]
for i, (name, desc, note) in enumerate(arch):
    yy = y + i*56
    R(X0, yy, WD, 48, CARD2 if i%2==0 else CARD, 4)
    R(X0+8, yy+10, 54, 28, ACID if i==1 else BG, 3)
    T(X0+35, yy+17, name, 8.5, BG if i==1 else ACID, True, anchor="ma")
    T(X0+72, yy+9, desc, 7.8, WHITE)
    T(X0+72, yy+26, note, 6.5, DIM)
# 箭头
for i in range(3):
    yy = y + i*56 + 50
    T(X0+WD/2, yy, "▼", 6, ACID, anchor="ma")
y += 4*56 + 8

# ============ 03 核心效果 (620-940) ============
T(X0, y, "03", 16, ACID, True, True)
T(X0+22, y+3, "核心效果：分布决定方法", 13, WHITE, True)
y += 28
# 上半：口语体 vs 病历体
half = (WD - 16) / 2
for side in range(2):
    bx = X0 + side*(half+16)
    R(bx, y, half, 150, CARD, 4)
    if side == 0:
        T(bx+12, y+8, "口语体输入", 9, GOLD, True)
        T(bx+12, y+22, "教材 37 例", 6.5, DIM)
        bars = [("TF-IDF 检索", 0.919, ACID), ("轻量分类器", 0.054, DIM)]
    else:
        T(bx+12, y+8, "病历体输入", 9, GOLD, True)
        T(bx+12, y+22, "TCM-SD 683 例", 6.5, DIM)
        bars = [("轻量分类器", 0.929, ACID), ("TF-IDF 检索", 0.127, DIM)]
    by = y + 40
    for label, val, col in bars:
        T(bx+12, by, label, 7, GREY)
        bw = half - 92
        R(bx+12, by+12, int(bw*val), 9, col, 2)
        T(bx+12+int(bw*val)+6, by+10, "%.1f%%" % (val*100), 8, col, True)
        by += 30
    T(bx+12, y+120, ("→ 口语输入走检索" if side==0 else "→ 病历输入走分类器"), 8, WHITE, True)
    T(bx+12, y+134, ("TF-IDF 91.9% vs 分类器 5.4%" if side==0 else "分类器 92.9% vs TF-IDF 12.7%"), 6.2, GREY)
y += 158
# 关键指标条
keys = [("风格判定", "99.95%", "可解释判据"), ("拒答线", "0.50", "该档准确率仅 55%"), ("分类器覆盖", "16/37", "其余由检索补充")]
kw = (WD - 24) / 3
for i, (name, val, note) in enumerate(keys):
    kx = X0 + i*(kw+12)
    R(kx, y, kw, 52, CARD2, 4)
    T(kx+12, y+7, name, 7, GREY)
    T(kx+12, y+18, val, 15, ACID, True)
    T(kx+12, y+39, note, 5.8, DIM)
y += 66

# ============ 04 大模型对比 (940-1060) ============
T(X0, y, "04", 16, ACID, True, True)
T(X0+22, y+3, "与大模型零样本对比（同一 683 例测试集）", 13, WHITE, True)
y += 28
R(X0, y, WD, 76, CARD, 4)
cols = [("本作品分类器", "92.9", ACID, "领先多数类基线 +15.7 pp"),
        ("DeepSeek 零样本", "75.8", WHITE, "未超过最笨基线"),
        ("多数类基线", "77.3", DIM, "永远答样本量最大类")]
cw = WD / 3
for i, (name, val, col, note) in enumerate(cols):
    cx = X0 + i*cw + 16
    T(cx, y+8, name, 6.8, GREY)
    T(cx, y+18, val + "%", 20, col, True)
    T(cx, y+52, note, 6.0, col if i==0 else DIM)
y += 84

# ============ 05 功能模块 (1060-1210) ============
T(X0, y, "05", 16, ACID, True, True)
T(X0+22, y+3, "六大功能模块", 13, WHITE, True)
y += 28
mods = [("知识检索", "五种方法横向对比，每条候选展示命中证据与未识别文本"),
        ("辨证训练", "从 54,152 条真实病历生成题目，答案锚定知识库"),
        ("相似鉴别", "2,556 组真实病历对比，共现差异精确到百分比"),
        ("方剂治法", "68 证型 · 314 条治法 · 215 首方剂，含干扰项辨析"),
        ("舌脉辨识", "24/72 证型（四诊合参，特征重叠者明确排除）"),
        ("学习报告", "统计易混证型对，形成练—评—补闭环")]
for i, (name, desc) in enumerate(mods):
    col = i % 2
    row = i // 2
    mx = X0 + col*(WD/2 + 8)
    my = y + row*38
    R(mx, my, WD/2 - 8, 33, CARD, 3)
    T(mx+10, my+4, name, 7.5, ACID, True)
    T(mx+10, my+16, desc, 5.9, GREY)
y += 3*38 + 8

# ============ 06 创新点 (1210-1420) ============
T(X0, y, "06", 16, ACID, True, True)
T(X0+22, y+3, "创新点", 13, WHITE, True)
y += 28
inno = [("① 分布决定方法", "量化方法适用性反转（91.9% vs 12.7%），用可解释判据实现输入风格路由"),
        ("② 不确定性可验证", "按实测准确率校准置信度；引入正确性探针；证据不足时拒答"),
        ("③ 生成与判定分离", "答案由知识库锚定，大模型只负责文字，程序负责校验与随机化"),
        ("④ 前沿方法适配验证", "将选择性预测与保形预测首次用于中医证型辨识，并报告失效边界")]
for i, (t, desc) in enumerate(inno):
    yy = y + i*36
    R(X0, yy, WD, 31, CARD, 3)
    d.rectangle([px(X0), px(yy), px(X0+3.5), px(yy+31)], fill=ACID)
    T(X0+12, yy+4, t, 8, WHITE, True)
    T(X0+12, yy+17, desc, 6.2, GREY)
y += 4*36 + 6

# ============ 07 数据规模 (1420-1510) ============
T(X0, y, "07", 16, ACID, True, True)
T(X0+22, y+3, "数据规模与可复现性", 13, WHITE, True)
y += 28
stats = [("54,152", "真实电子病历", "TCM-SD · CC-BY-SA"), ("1,027", "证型知识条目", "结构化知识库"),
         ("2,556", "相似证型对比对", "共现实测差异"), ("151", "题库题目", "核心校验 100% 通过")]
sw = (WD - 3*10) / 4
for i, (num, label, note) in enumerate(stats):
    sx = X0 + i*(sw+10)
    R(sx, y, sw, 58, CARD, 4)
    T(sx+10, y+6, num, 13, ACID, True)
    T(sx+10, y+28, label, 6.5, WHITE)
    T(sx+10, y+40, note, 5.4, DIM)
y += 62
# 复现说明条
R(X0, y, WD, 30, CARD2, 3)
T(X0+12, y+5, "11 个复现脚本 · 全部指标可独立复现", 7.5, ACID, True)
T(X0+12, y+17, "评测口径、去泄漏规则、置信区间与统计检验方法完整公开；方法论局限主动披露", 6.0, GREY)
y += 40

# ============ 08 应用与部署 (1510-1570) ============
T(X0, y, "08", 16, ACID, True, True)
T(X0+22, y+3, "应用场景与部署", 13, WHITE, True)
y += 28
R(X0, y, WD, 36, CARD2, 4)
T(X0+12, y+5, "课堂演示 → 课后自查 → 科研候选标注 → 横向迁移方剂学 / 针灸学（约 1 人日/学科）", 6.8, GREY)
T(X0+12, y+20, "完全离线 · 无需 GPU · 模型 7.6 MB · 单条推理 0.55 ms · 普通笔记本可部署", 6.8, ACID, True)

# ============ 底部 ============
HL(X0, 1480, WD, DIM, 0.4)
T(X0, 1486, "本草辨证 · 第八届 AIC 全球校园人工智能算法精英大赛 · 算法创新赛 · AI+学科交叉", 5.3, DIM)
T(X0, 1494, "非医疗器械 · 不提供临床诊断 · 输出定位为教学与科研辅助", 4.9, DIM)

img.save(OUT, "PNG")
print("已生成:", OUT, "%d x %d px (%.0f x %.0f cm)" % (W_PX, H_PX, MM_W/10, MM_H/10))