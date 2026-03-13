**NIfTI-to-HTML Evaluator**

A zero-backend utility converting NIfTI volumes into a responsive, standalone HTML/JS viewer. Optimized for rapid blinded evaluation of medical image synthesis / segmentation.

## **Technical Core**

* **Architecture:** Generates a static HTML/CSS/JS bundle. Zero-latency rendering via client-side logic.  
* **State Management:** Persistence via browser localStorage API. Supports session recovery and progress tracking without a database.  
* **Slicing Engine:** ROI-aware extraction. Uses segmentation boundaries to determine optimal Z-axis context.  
* **Blinding:** Automated directory shuffling with a generated evaluation\_key.csv for de-identification.

## **Controls & UX**

* **Axial Navigation:** Mouse Wheel to increment/decrement slice index.  
* **Comparison (The "Blink" Test):** Ctrl \+ Wheel (or Shift \+ Wheel) for real-to-synthetic opacity crossfading.  
* **Toggle:** T key for instant 0/100% overlay flip.  
* **Data Collection:** Integrated 1-5 rating buttons. Supports Auto-Next patient for high-throughput evaluation.  
* **Portability:** Export CSV / Load CSV for transferring evaluation data between machines.

## **Installation and usage**
```
pip install -r requirements.txt
````


```
python generate_pacs.py \\  
  \--base_dir brats_subset \\  
  \--ref t1c --seg seg \\  
  \--bg t1n t2f t2w \\  
  \--fake GAN UNet \\  
```

Files must share a common PatientID prefix across subfolders.
```
brats_subset/UNet/
Patient_001.nii.gz  Patient_002.nii.gz 

brats_subset/t1c/
Patient_001.nii.gz  Patient_002.nii.gz 
```