import os, uuid, time, base64, json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

app = Flask(__name__)
app.config['UPLOAD_FOLDER']  = 'static/uploads'
app.config['RESULT_FOLDER']  = 'static/results'
app.config['REPORT_FOLDER']  = 'static/reports'
ALLOWED = {'png','jpg','jpeg','bmp','tiff'}
for d in [app.config['UPLOAD_FOLDER'], app.config['RESULT_FOLDER'], app.config['REPORT_FOLDER']]:
    os.makedirs(d, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# ─────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────
class CBAM(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, ch//r), nn.ReLU(),
            nn.Linear(ch//r, ch), nn.Sigmoid())
        self.sa = nn.Sequential(nn.Conv2d(2,1,7,padding=3), nn.Sigmoid())
    def forward(self, x):
        x = x * self.ca(x).view(x.size(0),x.size(1),1,1)
        return x * self.sa(torch.cat([x.mean(1,keepdim=True),
                                      x.max(1,keepdim=True)[0]],1))

class DoubleConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(i,o,3,padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            nn.Conv2d(o,o,3,padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
    def forward(self, x): return self.b(x)

class UNetGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        b = 64
        self.e1=DoubleConv(3,b);    self.e2=DoubleConv(b,b*2)
        self.e3=DoubleConv(b*2,b*4); self.e4=DoubleConv(b*4,b*8)
        self.pool=nn.MaxPool2d(2)
        self.bottleneck=nn.Sequential(DoubleConv(b*8,b*16), CBAM(b*16))
        self.up4=nn.ConvTranspose2d(b*16,b*8,2,stride=2); self.d4=DoubleConv(b*16,b*8)
        self.up3=nn.ConvTranspose2d(b*8,b*4,2,stride=2);  self.d3=DoubleConv(b*8,b*4)
        self.up2=nn.ConvTranspose2d(b*4,b*2,2,stride=2);  self.d2=DoubleConv(b*4,b*2)
        self.up1=nn.ConvTranspose2d(b*2,b,2,stride=2);    self.d1=DoubleConv(b*2,b)
        self.out=nn.Sequential(nn.Conv2d(b,3,1), nn.Sigmoid())
    def forward(self, x):
        s1=self.e1(x); s2=self.e2(self.pool(s1))
        s3=self.e3(self.pool(s2)); s4=self.e4(self.pool(s3))
        b=self.bottleneck(self.pool(s4))
        d4=self.d4(torch.cat([self.up4(b),s4],1))
        d3=self.d3(torch.cat([self.up3(d4),s3],1))
        d2=self.d2(torch.cat([self.up2(d3),s2],1))
        d1=self.d1(torch.cat([self.up1(d2),s1],1))
        return self.out(d1)

G = UNetGenerator().to(DEVICE)
if os.path.exists('models/generator.pth'):
    G.load_state_dict(torch.load('models/generator.pth', map_location=DEVICE))
    print('✓ Generator loaded')
G.eval()

TF = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])

# ─────────────────────────────────────────────
# PREPROCESSING PIPELINE
# ─────────────────────────────────────────────
def stage1_green_channel(img_bgr):
    """Stage 1: Extract green channel"""
    green = img_bgr[:,:,1]
    return cv2.merge([green, green, green])

def stage2_bg_normalisation(img_bgr):
    """Stage 2: Gaussian background normalisation (Novel)"""
    green = img_bgr[:,:,1].astype(np.float32)
    sigma = int(0.15 * min(green.shape)) | 1
    bg    = cv2.GaussianBlur(green, (sigma,sigma), 0)
    norm  = np.clip(green - bg + green.mean(), 0, 255).astype(np.uint8)
    return cv2.merge([norm, norm, norm])

def stage3_clahe(img_bgr):
    """Stage 3: CLAHE contrast enhancement"""
    green = img_bgr[:,:,1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enh   = clahe.apply(green)
    return cv2.merge([enh, enh, enh])

def stage4_denoising(img_bgr):
    """Stage 4: Bilateral denoising"""
    return cv2.bilateralFilter(img_bgr, 9, 75, 75)

def stage5_gamma(img_bgr):
    """Stage 5: Gamma correction"""
    lut = np.array([min(255,int((i/255)**(1/1.2)*255))
                    for i in range(256)], np.uint8)
    return cv2.LUT(img_bgr, lut)

def classical_preprocess(img_bgr):
    """Run all 5 classical stages"""
    out = stage1_green_channel(img_bgr)
    out = stage2_bg_normalisation(out)
    out = stage3_clahe(out)
    out = stage4_denoising(out)
    out = stage5_gamma(out)
    return out

def regan_enhance(img_bgr):
    """Stage 6: RE-GAN deep enhancement"""
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    inp = TF(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        enh_t = G(inp)
    enh_np = (enh_t.squeeze().permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
    return cv2.cvtColor(enh_np, cv2.COLOR_RGB2BGR)

# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────
def locate_optic_disc(img_bgr):
    h, w  = img_bgr.shape[:2]
    total = h * w
    red   = img_bgr[:,:,2].astype(np.float32)
    green = img_bgr[:,:,1].astype(np.float32)
    bright = ((red+green)/2).astype(np.uint8)
    blurred = cv2.GaussianBlur(bright, (21,21), 0)
    thresh_val = np.percentile(blurred, 85)
    _, mask = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_contour = None
    best_score   = -1
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 0.005*total or area > 0.25*total: continue
        perim = cv2.arcLength(c, True)
        if perim == 0: continue
        circ = 4*np.pi*area/(perim**2)
        tmp  = np.zeros((h,w), np.uint8)
        cv2.drawContours(tmp,[c],-1,255,-1)
        mb   = cv2.mean(bright, mask=tmp)[0]
        score= circ * mb
        if score > best_score:
            best_score   = score
            best_contour = c
    return best_contour

def extract_features(img_bgr):
    h, w  = img_bgr.shape[:2]
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    disc_contour = locate_optic_disc(img_bgr)

    if disc_contour is None:
        disc_cx, disc_cy = int(w*0.6), int(h*0.5)
        disc_r = int(min(h,w)*0.12)
        disc_H = disc_W = float(disc_r*2)
        disc_area = np.pi*disc_r*disc_r
    else:
        disc_area = cv2.contourArea(disc_contour)
        M = cv2.moments(disc_contour)
        disc_cx = int(M['m10']/max(M['m00'],1))
        disc_cy = int(M['m01']/max(M['m00'],1))
        if len(disc_contour) >= 5:
            (_,_),(disc_W,disc_H),_ = cv2.fitEllipse(disc_contour)
        else:
            disc_H = disc_W = 2*np.sqrt(disc_area/np.pi)

    disc_H = max(float(disc_H), 1.0)
    disc_W = max(float(disc_W), 1.0)
    disc_r = int((disc_H+disc_W)/4)

    # Cup detection within disc ROI
    margin = int(disc_r*1.2)
    x1=max(0,disc_cx-margin); x2=min(w,disc_cx+margin)
    y1=max(0,disc_cy-margin); y2=min(h,disc_cy+margin)
    roi = gray[y1:y2, x1:x2]

    cup_H = cup_W = float(disc_H)*0.4
    if roi.size > 0:
        p95 = np.percentile(roi, 95)
        p80 = np.percentile(roi, 80)
        ct  = int(p80+(p95-p80)*0.4)
        _,cm = cv2.threshold(roi, ct, 255, cv2.THRESH_BINARY)
        kn   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
        cm   = cv2.morphologyEx(cm, cv2.MORPH_CLOSE, kn)
        cs,_ = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cs:
            rcx,rcy = roi.shape[1]//2, roi.shape[0]//2
            def dtc(c):
                M2=cv2.moments(c)
                if M2['m00']==0: return 9999
                return np.sqrt((M2['m10']/M2['m00']-rcx)**2+(M2['m01']/M2['m00']-rcy)**2)
            cc = min(cs, key=dtc)
            if len(cc)>=5:
                (_,_),(cup_W,cup_H),_ = cv2.fitEllipse(cc)
                cup_H = max(float(cup_H),0.1)
                cup_W = max(float(cup_W),0.1)

    vCDR = min(cup_H/disc_H, 1.0)
    hCDR = min(cup_W/disc_W, 1.0)
    cup_area = np.pi*(cup_H/2)*(cup_W/2)
    RDAR = min(max(disc_area-cup_area,0)/max(disc_area,1), 1.0)

    # ISNT Rule
    dm = np.zeros((h,w), np.uint8)
    if disc_contour is not None:
        cv2.drawContours(dm,[disc_contour],-1,255,-1)
    else:
        cv2.circle(dm,(disc_cx,disc_cy),disc_r,255,-1)
    q = {'I':dm[disc_cy:,    disc_cx-w//8:disc_cx+w//8],
         'S':dm[:disc_cy,    disc_cx-w//8:disc_cx+w//8],
         'N':dm[disc_cy-h//8:disc_cy+h//8, disc_cx:],
         'T':dm[disc_cy-h//8:disc_cy+h//8, :disc_cx]}
    rt   = {k:float(np.sum(v>0)) for k,v in q.items()}
    isnt = sum([rt['I']>rt['S'],rt['S']>rt['N'],rt['N']>rt['T']])

    # Vessel density
    vm   = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, np.ones((9,9),np.uint8))
    _,vb = cv2.threshold(vm, 20, 255, cv2.THRESH_BINARY)
    vd   = float(np.sum(vb>0))/(h*w)

    # Disc boundary sharpness (image quality metric)
    gx   = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
    gy   = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
    sharpness = float(np.mean(np.sqrt(gx**2+gy**2)))

    return {
        'vCDR':            round(vCDR,3),
        'hCDR':            round(hCDR,3),
        'RDAR':            round(RDAR,3),
        'ISNT':            isnt,
        'vessel_density':  round(vd,4),
        'disc_H':          round(disc_H,1),
        'disc_W':          round(disc_W,1),
        'cup_H':           round(cup_H,1),
        'cup_W':           round(cup_W,1),
        'sharpness':       round(sharpness,2),
        'disc_cx':         disc_cx,
        'disc_cy':         disc_cy,
    }

def compute_risk(feats):
    vCDR = feats['vCDR']; RDAR = feats['RDAR']
    isnt = feats['ISNT']; vd   = feats['vessel_density']
    if   vCDR < 0.45: cs = 0.05
    elif vCDR < 0.55: cs = 0.20
    elif vCDR < 0.65: cs = 0.45
    elif vCDR < 0.75: cs = 0.70
    elif vCDR < 0.85: cs = 0.85
    else:              cs = 0.95
    if   RDAR > 0.65: rs = 0.05
    elif RDAR > 0.50: rs = 0.20
    elif RDAR > 0.35: rs = 0.50
    elif RDAR > 0.20: rs = 0.75
    else:              rs = 0.90
    isnts = (3-isnt)/3.0
    if   vd > 0.12: vs = 0.10
    elif vd > 0.08: vs = 0.30
    elif vd > 0.05: vs = 0.55
    else:            vs = 0.75
    return round((0.50*cs + 0.25*rs + 0.15*isnts + 0.10*vs)*100, 1)

def risk_label(p):
    if p < 35: return 'Low Risk',     '#22c55e', '✅', 'Optic nerve head appears within normal limits. Routine annual eye examination recommended.'
    if p < 60: return 'Moderate Risk','#f59e0b', '⚠️', 'Some features warrant attention. Ophthalmologist review with IOP measurement recommended.'
    return 'High Risk','#ef4444','🔴','Significant structural features consistent with glaucoma. Urgent ophthalmology referral recommended.'

# ─────────────────────────────────────────────
# PDF REPORT GENERATION
# ─────────────────────────────────────────────
def generate_pdf_report(uid, orig_path, classical_path, enhanced_path,
                         feats, prob, risk_lbl, risk_desc, t_classical, t_enhance):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, Image as RLImage,
                                    HRFlowable)
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

    report_path = os.path.join(app.config['REPORT_FOLDER'], f'{uid}_report.pdf')
    doc = SimpleDocTemplate(report_path, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    DARK  = colors.HexColor('#1B2A4A')
    MID   = colors.HexColor('#2E5FA3')
    LIGHT = colors.HexColor('#D6E4F7')
    GREEN = colors.HexColor('#22c55e')
    AMBER = colors.HexColor('#f59e0b')
    RED   = colors.HexColor('#ef4444')
    GREY  = colors.HexColor('#F4F6FA')

    risk_color = {'Low Risk':GREEN,'Moderate Risk':AMBER,'High Risk':RED}.get(risk_lbl, MID)

    title_style = ParagraphStyle('title', fontSize=18, fontName='Helvetica-Bold',
                                 textColor=DARK, spaceAfter=4, alignment=TA_CENTER)
    sub_style   = ParagraphStyle('sub', fontSize=11, fontName='Helvetica',
                                 textColor=colors.HexColor('#6B7A99'), spaceAfter=2, alignment=TA_CENTER)
    h2_style    = ParagraphStyle('h2', fontSize=13, fontName='Helvetica-Bold',
                                 textColor=MID, spaceBefore=14, spaceAfter=6)
    h3_style    = ParagraphStyle('h3', fontSize=11, fontName='Helvetica-Bold',
                                 textColor=DARK, spaceBefore=8, spaceAfter=4)
    body_style  = ParagraphStyle('body', fontSize=10, fontName='Helvetica',
                                 textColor=colors.black, spaceAfter=4,
                                 leading=14, alignment=TA_JUSTIFY)
    small_style = ParagraphStyle('small', fontSize=9, fontName='Helvetica',
                                 textColor=colors.HexColor('#6B7A99'), spaceAfter=2)

    story = []

    # ── Header ──
    story.append(Paragraph('GlaucoScan — RE-GAN Analysis Report', title_style))
    story.append(Paragraph('Fundus Image Preprocessing & Clinical Feature Extraction', sub_style))
    story.append(Paragraph(f'Generated: {datetime.now().strftime("%d %B %Y, %I:%M %p")}  |  Analysis ID: {uid}', small_style))
    story.append(HRFlowable(width='100%', thickness=2, color=MID, spaceAfter=12))

    # ── Risk Banner ──
    risk_table = Table([[
        Paragraph(f'{prob}%', ParagraphStyle('rp', fontSize=32, fontName='Helvetica-Bold',
                                              textColor=risk_color, alignment=TA_CENTER)),
        Paragraph(f'{risk_lbl}\n{risk_desc}',
                  ParagraphStyle('rl', fontSize=11, fontName='Helvetica',
                                 textColor=DARK, leading=16))
    ]], colWidths=[4*cm, 13*cm])
    risk_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), GREY),
        ('BOX',        (0,0), (-1,-1), 1.5, risk_color),
        ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING',(0,0),(-1,-1),10),
        ('LEFTPADDING',(0,0),(-1,-1),12),
    ]))
    story.append(risk_table)
    story.append(Spacer(1, 14))

    # ── Image comparison ──
    story.append(Paragraph('Preprocessing Results', h2_style))
    story.append(Paragraph(
        'The fundus image was processed through a 5-stage classical pipeline followed by RE-GAN '
        'deep learning enhancement. The three images below show the progression from raw input '
        'to fully enhanced output.', body_style))
    story.append(Spacer(1, 6))

    def safe_img(path, w, h):
        try:
            return RLImage(path, width=w*cm, height=h*cm)
        except:
            return Paragraph('[Image unavailable]', small_style)

    img_w, img_h = 5.5, 5.5
    img_table = Table([
        [safe_img(orig_path,img_w,img_h),
         safe_img(classical_path,img_w,img_h),
         safe_img(enhanced_path,img_w,img_h)],
        [Paragraph('Original\n(Raw input)', ParagraphStyle('ic',fontSize=9,fontName='Helvetica',alignment=TA_CENTER)),
         Paragraph('Classical Pipeline\n(Stages 1–5)', ParagraphStyle('ic',fontSize=9,fontName='Helvetica',alignment=TA_CENTER)),
         Paragraph('RE-GAN Enhanced\n(Deep learning)', ParagraphStyle('ic',fontSize=9,fontName='Helvetica-Bold',alignment=TA_CENTER,textColor=MID))],
    ], colWidths=[6*cm, 6*cm, 6*cm])
    img_table.setStyle(TableStyle([
        ('ALIGN',  (0,0),(-1,-1),'CENTER'),
        ('VALIGN', (0,0),(-1,-1),'MIDDLE'),
        ('TOPPADDING',(0,0),(-1,-1),4),
        ('BOTTOMPADDING',(0,0),(-1,-1),4),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 8))

    # Processing times
    pt = Table([[
        Paragraph(f'Classical pipeline: {t_classical} ms', small_style),
        Paragraph(f'RE-GAN enhancement: {t_enhance} ms', small_style),
        Paragraph(f'Total: {round(t_classical+t_enhance,1)} ms', small_style),
    ]], colWidths=[6*cm,6*cm,6*cm])
    story.append(pt)
    story.append(Spacer(1, 14))
    story.append(HRFlowable(width='100%', thickness=0.5, color=LIGHT, spaceAfter=10))

    # ── Clinical Features ──
    story.append(Paragraph('Extracted Clinical Features', h2_style))
    story.append(Paragraph(
        'The following five biomarkers were extracted from the RE-GAN-enhanced image using '
        'the two-stage optic disc localisation algorithm. All features are grounded in '
        'published ophthalmological literature.', body_style))
    story.append(Spacer(1, 8))

    def interp_cdr(v):
        if v < 0.45: return 'Normal'
        if v < 0.65: return 'Borderline'
        return 'Elevated (>0.65 threshold)'

    def interp_rdar(v):
        if v > 0.55: return 'Normal rim tissue'
        if v > 0.35: return 'Borderline rim loss'
        return 'Significant rim loss'

    def interp_isnt(v):
        return {3:'Full compliance (normal)',2:'Minor violation',
                1:'Moderate violation',0:'Complete violation'}[v]

    def interp_vd(v):
        if v > 0.12: return 'Normal vessel coverage'
        if v > 0.08: return 'Mildly reduced'
        return 'Reduced vessel density'

    feat_data = [
        ['Feature', 'Value', 'Clinical Range', 'Interpretation', 'Weight'],
        ['Vertical CDR', f'{feats["vCDR"]:.3f}', '< 0.65 normal', interp_cdr(feats["vCDR"]), '50%'],
        ['Horizontal CDR', f'{feats["hCDR"]:.3f}', '< 0.65 normal', interp_cdr(feats["hCDR"]), '—'],
        ['Rim-Disc Area Ratio', f'{feats["RDAR"]:.3f}', '> 0.50 normal', interp_rdar(feats["RDAR"]), '25%'],
        ['ISNT Score', f'{feats["ISNT"]} / 3', '3 = fully normal', interp_isnt(feats["ISNT"]), '15%'],
        ['Vessel Density', f'{feats["vessel_density"]*100:.2f}%', '> 10% normal', interp_vd(feats["vessel_density"]), '10%'],
    ]
    ft = Table(feat_data, colWidths=[4*cm,2.2*cm,3*cm,5*cm,1.8*cm])
    ft.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0), DARK),
        ('TEXTCOLOR', (0,0),(-1,0), colors.white),
        ('FONTNAME',  (0,0),(-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',  (0,0),(-1,0), 9),
        ('FONTNAME',  (0,1),(-1,-1),'Helvetica'),
        ('FONTSIZE',  (0,1),(-1,-1), 9),
        ('BACKGROUND',(0,1),(-1,-1), colors.white),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white, GREY]),
        ('BOX',  (0,0),(-1,-1), 0.5, colors.HexColor('#BDC8D8')),
        ('GRID', (0,0),(-1,-1), 0.5, colors.HexColor('#BDC8D8')),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('TOPPADDING',(0,0),(-1,-1),6),
        ('BOTTOMPADDING',(0,0),(-1,-1),6),
        ('LEFTPADDING',(0,0),(-1,-1),8),
    ]))
    story.append(ft)
    story.append(Spacer(1, 14))

    # ── Disc measurements ──
    story.append(Paragraph('Optic Disc & Cup Measurements', h2_style))
    disc_data = [
        ['Measurement', 'Value', 'Measurement', 'Value'],
        ['Disc Height (px)', f'{feats["disc_H"]:.1f}', 'Cup Height (px)', f'{feats["cup_H"]:.1f}'],
        ['Disc Width (px)',  f'{feats["disc_W"]:.1f}', 'Cup Width (px)',  f'{feats["cup_W"]:.1f}'],
        ['Disc Centre X',   str(feats["disc_cx"]),    'Image Sharpness', f'{feats["sharpness"]:.2f}'],
        ['Disc Centre Y',   str(feats["disc_cy"]),    'Enhancement',     'RE-GAN (U-Net+CBAM)'],
    ]
    dt = Table(disc_data, colWidths=[4.5*cm,3*cm,4.5*cm,4*cm])
    dt.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0), MID),
        ('TEXTCOLOR', (0,0),(-1,0), colors.white),
        ('FONTNAME',  (0,0),(-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',  (0,0),(-1,-1), 9),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white, GREY]),
        ('BOX',  (0,0),(-1,-1), 0.5, colors.HexColor('#BDC8D8')),
        ('GRID', (0,0),(-1,-1), 0.5, colors.HexColor('#BDC8D8')),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('TOPPADDING',(0,0),(-1,-1),6),
        ('BOTTOMPADDING',(0,0),(-1,-1),6),
        ('LEFTPADDING',(0,0),(-1,-1),8),
    ]))
    story.append(dt)
    story.append(Spacer(1, 14))

    # ── Risk scoring breakdown ──
    story.append(Paragraph('Glaucoma Risk Score Breakdown', h2_style))
    story.append(Paragraph(
        f'The overall glaucoma probability of {prob}% is computed using a '
        'weighted clinical scoring formula based on published ophthalmological thresholds. '
        'Each feature contributes independently to the final score.', body_style))
    story.append(Spacer(1,6))

    def cdr_score(v):
        if v<0.45: return 5
        elif v<0.55: return 20
        elif v<0.65: return 45
        elif v<0.75: return 70
        elif v<0.85: return 85
        return 95
    def rdar_score(v):
        if v>0.65: return 5
        elif v>0.50: return 20
        elif v>0.35: return 50
        elif v>0.20: return 75
        return 90

    cs = cdr_score(feats['vCDR'])
    rs = rdar_score(feats['RDAR'])
    is_s = int((3-feats['ISNT'])/3.0*100)
    vd = feats['vessel_density']
    vs = 10 if vd>0.12 else (30 if vd>0.08 else (55 if vd>0.05 else 75))

    score_data = [
        ['Feature', 'Raw Score', 'Weight', 'Contribution'],
        ['Vertical CDR',       f'{cs}%',   '50%', f'{round(cs*0.50,1)}%'],
        ['Rim-Disc Area Ratio',f'{rs}%',   '25%', f'{round(rs*0.25,1)}%'],
        ['ISNT Score',         f'{is_s}%', '15%', f'{round(is_s*0.15,1)}%'],
        ['Vessel Density',     f'{vs}%',   '10%', f'{round(vs*0.10,1)}%'],
        ['TOTAL',              '',         '100%', f'{prob}%'],
    ]
    st = Table(score_data, colWidths=[5*cm,3*cm,3*cm,5*cm])
    st.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0), DARK),
        ('TEXTCOLOR', (0,0),(-1,0), colors.white),
        ('FONTNAME',  (0,0),(-1,0), 'Helvetica-Bold'),
        ('BACKGROUND',(0,-1),(-1,-1), LIGHT),
        ('FONTNAME',  (0,-1),(-1,-1),'Helvetica-Bold'),
        ('FONTSIZE',  (0,0),(-1,-1), 9),
        ('ROWBACKGROUNDS',(0,1),(-1,-2),[colors.white, GREY]),
        ('BOX',  (0,0),(-1,-1), 0.5, colors.HexColor('#BDC8D8')),
        ('GRID', (0,0),(-1,-1), 0.5, colors.HexColor('#BDC8D8')),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('TOPPADDING',(0,0),(-1,-1),6),
        ('BOTTOMPADDING',(0,0),(-1,-1),6),
        ('LEFTPADDING',(0,0),(-1,-1),8),
    ]))
    story.append(st)
    story.append(Spacer(1,14))

    # ── Pipeline info ──
    story.append(HRFlowable(width='100%',thickness=0.5,color=LIGHT,spaceAfter=10))
    story.append(Paragraph('Preprocessing Pipeline Used', h2_style))
    pipe_data = [
        ['Stage','Operation','Purpose','Status'],
        ['1','Green Channel Extraction','Maximise disc/vessel contrast','✓ Applied'],
        ['2','Gaussian BG Normalisation','Remove non-uniform illumination (Novel)','✓ Applied'],
        ['3','CLAHE Enhancement','Boost local contrast','✓ Applied'],
        ['4','Bilateral Denoising','Edge-preserving noise removal','✓ Applied'],
        ['5','Gamma Correction','Brightness adjustment','✓ Applied'],
        ['6','RE-GAN Enhancement','Deep learning enhancement (U-Net+CBAM)','✓ Applied'],
    ]
    pit = Table(pipe_data, colWidths=[1.5*cm,4.5*cm,6*cm,3*cm])
    pit.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0), DARK),
        ('TEXTCOLOR', (0,0),(-1,0), colors.white),
        ('FONTNAME',  (0,0),(-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',  (0,0),(-1,-1), 9),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white, GREY]),
        ('BOX',  (0,0),(-1,-1), 0.5, colors.HexColor('#BDC8D8')),
        ('GRID', (0,0),(-1,-1), 0.5, colors.HexColor('#BDC8D8')),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('TOPPADDING',(0,0),(-1,-1),5),
        ('BOTTOMPADDING',(0,0),(-1,-1),5),
        ('LEFTPADDING',(0,0),(-1,-1),6),
    ]))
    story.append(pit)
    story.append(Spacer(1,14))

    # ── Disclaimer ──
    story.append(HRFlowable(width='100%',thickness=0.5,color=LIGHT,spaceAfter=8))
    story.append(Paragraph('Important Notice', h3_style))
    story.append(Paragraph(
        'This report is generated by an automated research system (GlaucoScan RE-GAN) '
        'developed as part of a PhD research programme. It is intended for research and '
        'educational purposes only and does NOT constitute a medical diagnosis. '
        'All findings must be confirmed by a qualified ophthalmologist before any '
        'clinical decision is made. If you are experiencing vision problems, please '
        'consult a healthcare professional immediately.', body_style))
    story.append(Spacer(1,8))
    story.append(Paragraph(
        f'System: GlaucoScan RE-GAN v2.0  |  Dataset: DRISHTI-GS  |  '
        f'Device: {str(DEVICE).upper()}  |  '
        f'Report ID: {uid}', small_style))

    doc.build(story)
    return report_path

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error':'No file uploaded'}), 400
    f   = request.files['image']
    ext = f.filename.rsplit('.',1)[-1].lower()
    if ext not in ALLOWED:
        return jsonify({'error':'Invalid file type. Use PNG, JPG or BMP.'}), 400

    uid       = str(uuid.uuid4())[:8]
    orig_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{uid}_orig.png')
    f.save(orig_path)

    # Load & resize
    pil     = Image.open(orig_path).convert('RGB')
    img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    img_bgr = cv2.resize(img_bgr, (512,512))
    cv2.imwrite(orig_path, img_bgr)

    # Classical preprocessing
    t0  = time.time()
    cls = classical_preprocess(img_bgr)
    t_cl= round((time.time()-t0)*1000, 1)

    # Save classical result
    cl_path = os.path.join(app.config['RESULT_FOLDER'], f'{uid}_classical.png')
    cv2.imwrite(cl_path, cls)

    # RE-GAN enhancement
    t1  = time.time()
    enh = regan_enhance(cls)
    t_en= round((time.time()-t1)*1000, 1)

    # Save enhanced result
    enh_path = os.path.join(app.config['RESULT_FOLDER'], f'{uid}_enhanced.png')
    cv2.imwrite(enh_path, enh)

    # Feature extraction from enhanced image
    feats = extract_features(enh)
    prob  = compute_risk(feats)
    lbl, col, ico, desc = risk_label(prob)

    # Generate PDF report
    try:
        report_path = generate_pdf_report(
            uid, orig_path, cl_path, enh_path,
            feats, prob, lbl, desc, t_cl, t_en)
        report_url  = f'/download_report/{uid}'
    except Exception as e:
        print(f'PDF error: {e}')
        report_url = None

    def b64(path):
        with open(path,'rb') as fh:
            return 'data:image/png;base64,' + base64.b64encode(fh.read()).decode()

    return jsonify({
        'orig_img':      b64(orig_path),
        'classical_img': b64(cl_path),
        'enhanced_img':  b64(enh_path),
        'features':      feats,
        'glaucoma_prob': prob,
        'risk_label':    lbl,
        'risk_color':    col,
        'risk_icon':     ico,
        'risk_desc':     desc,
        't_classical':   t_cl,
        't_enhance':     t_en,
        'device':        str(DEVICE),
        'report_url':    report_url,
        'uid':           uid,
    })

@app.route('/download_report/<uid>')
def download_report(uid):
    path = os.path.join(app.config['REPORT_FOLDER'], f'{uid}_report.pdf')
    if not os.path.exists(path):
        return 'Report not found', 404
    return send_file(path, as_attachment=True,
                     download_name=f'GlaucoScan_Report_{uid}.pdf',
                     mimetype='application/pdf')

if __name__ == '__main__':
    print(f'✓ GlaucoScan v2 running on {DEVICE}')
    app.run(debug=False, host='0.0.0.0', port=5000)
