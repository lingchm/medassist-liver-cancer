# MedAssist-Liver: an AI-powered Liver Tumor Segmentation Tool

Liver cancer is a significant global health concern. Automated liver tumor segmentation remains to be a challenging tasks that demands more research. Read more about the background [here](documentation/research.md). 

This tool is designed to assist in the identification, segmentation, and analysis of lung and tumor from medical images. By uploading a CT scan image, a pre-trained machine learning model automatically segments the lung and tumor regions. The segmented tumor's characteristics such as shape, size, and location are then analyzed to produce an AI-generated diagnosis report of the lung cancer.

* Public link to the tool: [https://lingchmao-medassist-liver-cancer.hf.space](https://lingchmao-medassist-liver-cancer.hf.space) 
* How to setup and run locally: [documentation/user_manual](documentation/user_manual.md)
* Read more the deep learning model details in: [documentation/methodology](documentation/methodology.md)

## Main Features

This tool consists of two main features:

* **Liver and tumor segmentation**. Given a 3D CT scan, two pre-trained deep learning model are run sequentially, one generating segmentation of liver and the other generaitng segmentation of tumor within the liver. These models were developed in-house using the [HCC-TASE-Seg dataset](https://www.cancerimagingarchive.net/collection/hcc-tace-seg/). Model training/inference can be run with ```notebooks/run_best_model_notebook.ipynb```. 

* **AI-generated summary report**. Given the segmented liver and tumor masks, the model extracts tumor descriptors including leiosn size (cm), lesion, shape, lesion density, and involvement of adjacent organs. These information are given to a GPT model to generate a summary of lung cancer diagnosis.

![app flow](documentation/images/app.png)


## Cite
```
@article{wang2024livertumor,
  title = {A Holistic Weakly Supervised Approach for Liver Tumor Segmentation with Clinical Knowledge-Informed Label Smoothing},
  author = {Wang, Hairong and Mao, Lingchao and Zhang, Zihan and Li, Jing},
  journal = {IISE Transactions on Healthcare Systems Engineering (in preparation)},
}
```



