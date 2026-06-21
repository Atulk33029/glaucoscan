# GlaucoScan v2 — Local + Cloud Deployment Guide

## Project Structure
```
glaucoscan_deploy/
├── app.py                  ← Flask backend
├── templates/
│   └── index.html          ← Frontend UI
├── static/
│   ├── uploads/            ← Uploaded images (auto)
│   ├── results/            ← Enhanced images (auto)
│   └── reports/            ← PDF reports (auto)
├── models/
│   ├── generator.pth       ← Trained RE-GAN weights (download from Drive)
│   └── classifier.pth      ← Trained classifier weights (download from Drive)
├── requirements.txt
├── Procfile                ← For Render.com hosting
├── render.yaml             ← For Render.com hosting
├── setup_windows.bat       ← Windows one-click setup
└── run_windows.bat         ← Windows one-click run
```

---

## Part A: Train Model in Colab (one time only)

1. Open `GlaucoScan_DRISHTI_GS.ipynb` in Google Colab
2. Set runtime to T4 GPU
3. Run all training cells
4. Run `colab_save_weights.py` content to save weights to Drive
5. Download `generator.pth` and `classifier.pth` from Drive
6. Place them in `glaucoscan_deploy/models/`

---

## Part B: Run Locally on Windows

### First time setup:
1. Install Python from https://www.python.org/downloads/
   - IMPORTANT: Check "Add Python to PATH"
2. Double-click `setup_windows.bat`
3. Wait 5 minutes for installation

### Every time to run:
1. Double-click `run_windows.bat`
2. Open http://localhost:5000 in browser
3. Done!

---

## Part C: Host Online Free (Render.com)

### Step 1: Push to GitHub
1. Create account at https://github.com
2. Create new repository called `glaucoscan`
3. Upload all files from this folder to the repository
4. NOTE: Do NOT upload .pth weight files (too large for GitHub)

### Step 2: Deploy on Render.com
1. Create account at https://render.com
2. Click "New" → "Web Service"
3. Connect your GitHub repository
4. Settings:
   - Name: glaucoscan
   - Environment: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
5. Click "Create Web Service"
6. Wait 5-10 minutes for deployment
7. Your app will be live at: https://glaucoscan.onrender.com

### Step 3: Upload weights to Render
Since .pth files are too large for GitHub, use Render's disk:
1. In Render dashboard → your service → "Disks"
2. Add disk, mount path: `/opt/render/project/src/models`
3. SSH into Render and upload weights, OR
4. Use Google Drive direct download in app startup

### Alternative: Use the app without trained weights
The app works without weights — it just uses an untrained generator
which still applies classical preprocessing and extracts features correctly.
Only the RE-GAN enhancement step will be less effective.

---

## Environment Variables (optional)
Set these in Render dashboard if needed:
- `PORT` = 5000
- `FLASK_ENV` = production

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `python not found` | Reinstall Python, check "Add to PATH" |
| `pip install fails` | Run as Administrator |
| `port 5000 in use` | Kill other processes or change port in app.py |
| `model not loading` | Check models/ folder has .pth files |
| Render build fails | Check requirements.txt has correct versions |
