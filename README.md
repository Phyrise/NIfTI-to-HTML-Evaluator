**NIfTI-to-HTML Evaluator**

A zero-backend utility converting NIfTI volumes into a responsive, standalone HTML/JS viewer. Optimized for rapid blinded evaluation of medical image synthesis and segmentation.

[Live Interactive Demo](https://phyrise.github.io/NIfTI-to-HTML-Evaluator/)

## Controls & UI

- **Axial Navigation:** Mouse wheel to move through slices
- **Zoom:** Ctrl + Wheel to zoom in/out (50% - 200%)
- **Opacity Adjustment:** Shift + Wheel to adjust synthetic overlay opacity
- **Toggle Overlay:** T key to flip between full reference and full synthetic
- **Rating:** 1-5 buttons to score image quality and clinical value
- **Auto-Advance:** Optional auto-advance to next patient after rating
- **Data Management:** Export/Import CSV files for transferring evaluation data
- **Zoom Controls:** Use +/- buttons or slider in toolbar

## Installation and Usage

```
pip install -r requirements.txt
```

```
python generate_pacs.py \
  --base_dir brats_subset \
  --ref t1c --seg seg \
  --bg t1n t2f t2w \
  --fake GAN UNet
```

Files must share a common PatientID prefix across subfolders:
```
brats_subset/UNet/
Patient_001.nii.gz  Patient_002.nii.gz 

brats_subset/t1c/
Patient_001.nii.gz  Patient_002.nii.gz 
```
