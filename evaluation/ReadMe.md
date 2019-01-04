Evaluation

Usage: 

`python evaluate_single_submission.py #your prediction folder# #your gt_folder# --output_path=#your output table path# --export_table=True`

Structure:

* prediction folder:
----(prediction folder)  
    ----segmentation  
    ------------(prediction images here)  
* gt_folder:  
    ----Disc_Cup_Masks  
    ------------(ground truth images here)

Every name of prediction images should be found in ground truth images
