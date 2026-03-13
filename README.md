**NIfTI-to-HTML Viewer for Image Evaluation**

A Python-based utility that extracts 2D PNG slices from NIfTI volumes (current example: BraTS 3D MRI) and generates a standalone index.html viewer. This tool is designed for fast and localized comparison of medical image synthesis and segmentation models without a backend server.

## **Functional Specifications**

* **Execution Environment:** Generates a portable HTML/JS/CSS bundle. All image rendering and logic occur client-side.  
* **Data Persistence:** User ratings and evaluation data are stored in the browser's localStorage API. No data is transmitted over a network.  
* **Slice Selection:** Uses a stride-based algorithm that centers on detected ROI/segmentation boundaries to determine slice indices.  
* **Blinding Mechanism:** Model directories are shuffled during the generation process. A evaluation\_key.csv file is created to map shuffled IDs back to the original source directories.

## **Navigation & Controls**

| Action | Control | Result |
| :---- | :---- | :---- |
| **Slice Navigation** | Mouse Wheel | Increments/Decrements the Z-axis slice index. |
| **Layer Comparison** | Ctrl \+ Mouse Wheel | Adjusts the global opacity of the overlay/model layer. |
| **Rating** | Selection Buttons | Saves integer values to localStorage linked to the current case ID. |

## **Requirements**
```
pip install -r requirements.txt
```
## **Directory Structure**

Input data must follow a consistent naming convention across directories. If patient\_001.nii.gz exists in the reference folder, it must also exist in the segmentation and model folders to be processed.

input\_data/  
├── ref/             --- Reference/Ground Truth volumes  
├── segs/             --- ROI masks for slice centering  
├── bg\_t1n/          --- (Optional) Structural background volumes  
├── bg\_t2f/          --- (Optional) Structural background volumes  
├── model\_A/      --- Output from Model A  
└── model\_B/       --- Output from Model B

## **Usage**

Generate the PACS viewer by defining the source directories and the number of slices to extract per volume.

```
python generate\_pacs.py \\  
  \--ref_dir input_data/ref \\  
  \--seg_dir input_data/segs \\  
  \--bg_dirs input_data/bg_t1n input_data/bg_t2f \\  
  \--fake_dirs input_data/model_unet input_data/model_gan \\  
  \--out_dir pacs_demo \\  
  \--num_slices 10
```