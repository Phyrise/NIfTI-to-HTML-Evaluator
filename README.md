**NIfTI-to-HTML Evaluator**

A zero-backend utility converting NIfTI volumes into a responsive, standalone HTML/JS viewer. Optimized for rapid blinded evaluation of medical image synthesis and segmentation.

[Live Interactive Demo](https://phyrise.github.io/NIfTI-to-HTML-Evaluator/)

## Controls & UI

- **Axial Navigation:** Mouse wheel to move through slices
- **Zoom:** Double-click to zoom in/out, or use Ctrl+Wheel for smooth zooming
- **Comparison Mode:** 
  - Ctrl+Wheel: Adjust opacity between reference and synthetic images
  - T key: Toggle full overlay (0/100%)
- **Rating:** 1-5 star buttons to score image quality
- **Auto-Advance:** Optional auto-advance to next patient after rating
- **Data Management:** Export/Import CSV files for transferring evaluation data

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
