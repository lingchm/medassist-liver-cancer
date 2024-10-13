# User Instructions Manual

This tool is designed to assist in the identification, segmentation, and analysis of lung and tumor from medical images. By uploading a CT scan image, a pre-trained machine learning model will automatically segment the lung and tumor regions. Segmented tumor's characteristics such as shape, size, and location are then analyzed to produce an AI-generated diagnosis report of the lung cancer.


## Deployment

### Run locally

To locally deploy this tool, you need to follow these steps:
1. (optional) Setup a local environment. This can be done in multiple ways. For example, via Python's [venv](https://realpython.com/python-virtual-environments-a-primer/) module:
```python -m venv .venv```
```source .venv/bin/activate```

2. Install required dependencies. Note that Python>=3.11 is required. 
```pip install -r requirements.txt ```

3. Launch the app by running. Your app will be running on local URL http://127.0.0.1:7860. 
```python app.py``` 

If you want to create a shareable link, you can edit the following line in app.py:
```app.launch(share=True)```


### Deploy through HF Hosting

Hugging Faces (HF) has a nice integration with Gradio. This tool has been deployed through HF Spaces by running the following steps:

1. Create a free HF account
2. Create a repo in HF Spaces for this project
3. Git clone this repo in a local directory
4. (Option 1) From the terminal and at the project directory run
```gradio deploy```
5. (Option 2) From the browser, simply drag and drop all directory files including the requirements.txt. This will automatically deploy the model.


### Deploy through Github Pages

1. Create a Github Pages
2. Add the following embed to the HTML file:
```
<script
	type="module"
	src="https://gradio.s3-us-west-2.amazonaws.com/4.27.0/gradio.js"
></script>
```


## Quick Start

Using this tool consists of four simple steps: 

1. **Upload a CT scan image**. Currently we only accept files in .nrrd format. Alternatively, the user can select one of the example images from the bottom of the page. Once selected, click 'Upload' button. The image will be displayed in the image viewer. 

2. **Generate segmentation**. In the multi-choice menu, select either 'tumor' and/or 'liver' to generate tumor and/or liver segmentations, Click 'Generate Segmentation' to run inference using the pretrained deep learning model. This may take a few minutes. 

3. **Visualize and download segmentations**. Once completed, the segmented masks can be visualized in the image viewer overlaid on the images. The user can use the slicer to view different z-axis slices. The buttons 'Download liver mask' and 'Download tumor mask' can be used to export the segmentations as .npy files.

4. **Generate AI diagnostic report**. Lastly, clicking the 'Generate summary' button can generate an GPT-generated summary report for the image. 

