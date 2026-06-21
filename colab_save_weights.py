"""
Run this cell in Colab AFTER training is complete.
It saves your trained weights to Google Drive so you can
download them to your local Windows machine.
"""

import shutil, os

# Save to Google Drive
SAVE_DIR = '/content/drive/MyDrive/GlaucoScan_Weights'
os.makedirs(SAVE_DIR, exist_ok=True)

# Copy weights
for f in ['generator.pth', 'classifier.pth']:
    src = f'/content/glaucoscan/models/{f}'
    dst = os.path.join(SAVE_DIR, f)
    if os.path.exists(src):
        shutil.copy(src, dst)
        size = os.path.getsize(dst)/1024**2
        print(f'✓ Saved {f} ({size:.1f} MB) → {dst}')
    else:
        print(f'✗ Not found: {src}')

print(f'\n✓ Done! Go to Google Drive → GlaucoScan_Weights/')
print('  Download both .pth files to your PC')
print('  Place them in: glaucoscan_deploy/models/')
