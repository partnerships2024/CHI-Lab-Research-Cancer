# 1. Create the folder structure
mkdir -p PanTS-Segmentation/{data/images,data/masks,checkpoints,results}

# 2. Save README.md, train.py, test.py into PanTS-Segmentation/

# 3. Install dependencies
cd PanTS-Segmentation
pip install -r requirements.txt

# 4. Add your data
# Put images in data/images/ and masks in data/masks/
# Masks must be named <image_name>_mask.png

# 5. Train (creates checkpoints/model.h5)
python train.py

# 6. Test (creates results/*)
python test.py