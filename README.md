# MedAssist-Liver: an AI-powered Liver Tumor Segmentation Tool

This tool is designed to assist in the identification, segmentation, and analysis of lung and tumor from medical images. By uploading a CT scan image, a pre-trained machine learning model will automatically segment the lung and tumor regions. Segmented tumor's characteristics such as shape, size, and location are then analyzed to produce an AI-generated diagnosis report of the lung cancer.

This tool has been deployed through Hugging Face Spaces: [https://lingchmao-medassist-liver-cancer.hf.space](https://lingchmao-medassist-liver-cancer.hf.space) 

Model checkpoints and examples: [HF repo](https://huggingface.co/spaces/lingchmao/medassist-liver-cancer/tree/main)


## Quick Start

Using this tool consists of four simple steps: 

1. **Upload a CT scan image**. Currently we only accept files in .nrrd format. Alternatively, the user can select one of the example images from the bottom of the page. Once selected, click 'Upload' button. The image will be displayed in the image viewer. 

2. **Generate segmentation**. In the multi-choice menu, select either 'tumor' and/or 'liver' to generate tumor and/or liver segmentations, Click 'Generate Segmentation' to run inference using the pretrained deep learning model. This may take a few minutes. 

3. **Visualize and download segmentations**. Once completed, the segmented masks can be visualized in the image viewer overlaid on the images. The user can use the slicer to view different z-axis slices. The buttons 'Download liver mask' and 'Download tumor mask' can be used to export the segmentations as .npy files.

4. **Generate AI diagnostic report**. Lastly, clicking the 'Generate summary' button can generate an GPT-generated summary report for the image. 


## Main Features

This tool consists of two main features:

* **Liver and tumor segmentation**. Given a 3D CT scan, two pre-trained deep learning model are run sequentially, one generating segmentation of liver and the other generaitng segmentation of tumor within the liver. These models were developed in-house using ~100 patients from the [HCC-TASE-Seg dataset](https://www.cancerimagingarchive.net/collection/hcc-tace-seg/). The models achieved 0.954 dice score for lung segmentation and 0.570 dice score for tumor segmentation. The source code of model development can be found under ```model_development/run_best_model_notebook.ipynb```. 

* **AI-generated summary report**. Given the segmented liver and tumor masks, the model extracts tumor descriptors including leiosn size (cm), lesion, shape, lesion density, and involvement of adjacent organs. These information are given to a GPT model to generate a summary of lung cancer diagnosis.



