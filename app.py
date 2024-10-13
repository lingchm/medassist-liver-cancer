import os
import numpy as np
import gradio as gr
import torch
import monai 
import morphsnakes as ms
from src.sliding_window import sw_inference
from src.tumor_features import generate_features
from monai.networks.nets import SegResNetVAE
from monai.transforms import (
    LoadImage, Orientation, Compose, ToTensor, Activations, 
    FillHoles, KeepLargestConnectedComponent, AsDiscrete, ScaleIntensityRange
)
import llama_cpp
import llama_cpp.llama_tokenizer


# global params 
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
examples_path = [
    os.path.join(THIS_DIR, 'examples', 'HCC_003.nrrd'),
    os.path.join(THIS_DIR, 'examples', 'HCC_007.nrrd'),
    os.path.join(THIS_DIR, 'examples', 'HCC_018.nrrd')
]
models_path = {
    "liver": os.path.join(THIS_DIR, 'checkpoints', 'liver_3DSegResNetVAE.pth'),
    "tumor": os.path.join(THIS_DIR, 'checkpoints', 'tumor_3DSegResNetVAE.pth') 
}
cache_path = {
    "liver mask": "liver_mask.npy",
    "tumor mask": "tumor_mask.npy"
}
device = "cpu"
mydict = {}


def render(image_name, x, selected_slice):
    
    if not isinstance(image_name, str) or '/' in image_name:
        image_name = image_name.name.split('/')[-1].replace(".nrrd","")
        
    if 'img' not in mydict[image_name].keys():
        return (np.zeros((512, 512)), []), f'z-value: {x}, (zmin: {None}, zmax: {None})'
    
    # set slider ranges 
    zmin, zmax = 0, mydict[image_name]['img'].shape[-1] - 1
    if x > zmax: x = zmax
    if x < zmin: x = zmin
    
    # image 
    img = mydict[image_name]['img'][:,:,x]
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) # scale to 0 and 1
    
    # masks
    annotations = []
    if 'liver mask' in mydict[image_name].keys():
        annotations.append((mydict[image_name]['liver mask'][:,:,x], "segmented liver"))
    if 'tumor mask' in mydict[image_name].keys(): 
        annotations.append((mydict[image_name]['tumor mask'][:,:,x], "segmented tumor"))
    
    return img, annotations


def load_liver_model():

    liver_model = SegResNetVAE(
        input_image_size=(512,512,16),
        vae_estimate_std=False, 
        vae_default_std=0.3, 
        vae_nz=256, 
        spatial_dims=3,
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=1,
        norm='instance',
        out_channels=2,
        dropout_prob=0.2,
    )
    
    liver_model.load_state_dict(torch.load(models_path['liver'], map_location=torch.device(device)))
    
    return liver_model 


def load_tumor_model():

    tumor_model = SegResNetVAE(
            input_image_size=(256,256,32),
            vae_estimate_std=False, 
            vae_default_std=0.3, 
            vae_nz=256, 
            spatial_dims=3,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            in_channels=1,
            norm='instance',
            out_channels=3,
            dropout_prob=0.2,
        )
        
    tumor_model.load_state_dict(torch.load(models_path['tumor'], map_location=torch.device('cpu')))

    return tumor_model


def load_image(image, slider, selected_slice):
    
    global mydict
    
    image_name = image.name.split('/')[-1].replace(".nrrd","")
    mydict = {image_name: {}}

    preprocessing_liver = Compose([
        # load image 
        LoadImage(reader="NrrdReader", ensure_channel_first=True),
        # ensure orientation 
        Orientation(axcodes="PLI"),
        # convert to tensor 
        ToTensor()
    ])
    
    input = preprocessing_liver(image.name)
    mydict[image_name]["img"] = input[0].numpy() # first channel 
    
    print("Loaded image", image_name)
    
    image, annotations = render(image_name, slider, selected_slice)
    
    return f"😊 Your image is successfully loaded! Please use the slider to view the image (zmin: 1, zmax: {input.shape[-1]}).", (image, annotations)
    

def segment_tumor(image_name):
    
    if os.path.isfile(f"cache/{image_name}_{cache_path['tumor mask']}"):
        mydict[image_name]['tumor mask'] = np.load(f"cache/{image_name}_{cache_path['tumor mask']}")
    
    if 'tumor mask' in mydict[image_name].keys() and mydict[image_name]['tumor mask'] is not None:
        return 
    
    input = torch.from_numpy(mydict[image_name]['img'])
    
    tumor_model = load_tumor_model()
    
    preprocessing_tumor = Compose([
        ScaleIntensityRange(a_min=-200, a_max=250, b_min=0.0, b_max=1.0, clip=True)
    ])
    
    postprocessing_tumor = Compose([
        Activations(sigmoid=True),
        # Convert to binary predictions 
        AsDiscrete(argmax=True, to_onehot=3),
        # Remove small connected components for 1=liver and 2=tumor
        KeepLargestConnectedComponent(applied_labels=[2]),
        # Fill holes in the binary mask for 1=liver and 2=tumor
        FillHoles(applied_labels=[2]),
        ToTensor()
    ])
    
    # Preprocessing
    input = preprocessing_tumor(input)
    input = torch.multiply(input, torch.from_numpy(mydict[image_name]['liver mask'])) # mask non-liver regions 
        
    # Generate segmentation 
    with torch.no_grad():
        segmented_mask = sw_inference(tumor_model, input[None, None, :], (256,256,32), False, discard_second_output=True, overlap=0.2)[0] # input dimensions [B,C,H,W,Z]

    # Postprocess image 
    segmented_mask = postprocessing_tumor(segmented_mask)[-1].numpy() # background, liver, tumor
    segmented_mask = ms.morphological_chan_vese(segmented_mask, iterations=2, init_level_set=segmented_mask)
    segmented_mask = np.multiply(segmented_mask, mydict[image_name]['liver mask']) # Mask regions outside liver
    mydict[image_name]["tumor mask"] = segmented_mask
    
    # Saving
    np.save(f"cache/{image_name}_{cache_path['tumor mask']}", mydict[image_name]["tumor mask"])
    print(f"tumor mask saved to 'cache/{image_name}_{cache_path['tumor mask']}")
    
    return
    
    
def segment_liver(image_name):
    
    if os.path.isfile(f"cache/{image_name}_{cache_path['liver mask']}"):
        mydict[image_name]['liver mask'] = np.load(f"cache/{image_name}_{cache_path['liver mask']}")
    
    if 'liver mask' in mydict[image_name].keys() and mydict[image_name]['liver mask'] is not None:
        return 
    
    input = torch.from_numpy(mydict[image_name]['img'])
    
    # load model 
    liver_model = load_liver_model()
    
    # HU Windowing 
    preprocessing_liver = Compose([
        ScaleIntensityRange(a_min=-150, a_max=250, b_min=0.0, b_max=1.0, clip=True)
    ])
    
    postprocessing_liver = Compose([
          # Apply softmax activation to convert logits to probabilities
          Activations(sigmoid=True),
          # Convert predicted probabilities to discrete values (0 or 1)
          AsDiscrete(argmax=True, to_onehot=None),
          # Remove small connected components for 1=liver and 2=tumor
          KeepLargestConnectedComponent(applied_labels=[1]),
          # Fill holes in the binary mask for 1=liver and 2=tumor
          FillHoles(applied_labels=[1]),
          ToTensor()
    ])
    
    # Preprocessing
    input = preprocessing_liver(input)
 
    # Generate segmentation 
    with torch.no_grad():
        segmented_mask = sw_inference(liver_model, input[None, None, :], (512,512,16), False, discard_second_output=True, overlap=0.2)[0] # input dimensions [B,C,H,W,Z]

    # Postprocess image 
    segmented_mask = postprocessing_liver(segmented_mask)[0].numpy() # first channel 
    mydict[image_name]["liver mask"] = segmented_mask
    print(f"liver mask shape: {segmented_mask.shape}")
    
    # Saving
    np.save(f"cache/{image_name}_{cache_path['liver mask']}", mydict[image_name]["liver mask"])
    print(f"liver mask saved to cache/{image_name}_{cache_path['liver mask']}")
    
    return 


def segment(image, selected_mask, slider, selected_slice):
    
    image_name = image.name.split('/')[-1].replace(".nrrd", "")
    download_liver = gr.DownloadButton(label="Download liver mask", visible = False)
    download_tumor = gr.DownloadButton(label="Download tumor mask", visible = False)
    
    if 'liver mask' in selected_mask: 
        print('Segmenting liver...')
        segment_liver(image_name)
        download_liver = gr.DownloadButton(label="Download liver mask", value=f"cache/{image_name}_{cache_path['liver mask']}", visible=True)
    
    if 'tumor mask' in selected_mask:
        print('Segmenting tumor...')
        segment_tumor(image_name)
        download_tumor = gr.DownloadButton(label="Download tumor mask", value=f"cache/{image_name}_{cache_path['tumor mask']}", visible=True)
    
    image, annotations = render(image, slider, selected_slice)
    
    return f"🥳 Segmentation is completed. You can use the slider to view slices or proceed with generating a summary report.", download_liver, download_tumor, (image, annotations)


def generate_summary(image):
    
    image_name = image.name.split('/')[-1].replace(".nrrd","")
    
    if "liver mask" not in mydict[image_name] or "tumor mask" not in mydict[image_name]:
        return "⛔ You need to generate both liver and tumor masks before we can create a summary report.", "Not generated" 
    
    # extract tumor features from CT scan 
    features = generate_features(mydict[image_name]["img"], mydict[image_name]["liver mask"], mydict[image_name]["tumor mask"])
    print(features)
    
    # initialize LLM pulling from hugging face 
    llama = llama_cpp.Llama.from_pretrained(
        repo_id="Qwen/Qwen1.5-0.5B-Chat-GGUF",
        filename="*q8_0.gguf",
        tokenizer=llama_cpp.llama_tokenizer.LlamaHFTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B"),
        verbose=False
    )

    # openai.api_key = os.environ["OPENAI"]
    system_msg = """
    You are a radiologist. You use a segmentation model that extracts tumor characteristics from CT scans from which you generate a diagnosis report. 
    The report should include recommendations for next steps, and a disclaimer that these results should be taken with a grain of salt.
    """
    
    user_msg = f"""
    The tumor characteristics are:
    {str(features)}
    Please provide your interpretation of the findings and a differential diagnosis, considering the possibility of liver cancer (hepatocellular carcinoma or metastatic liver lesions).
    """
    print(user_msg)

    response = llama.create_chat_completion(
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.7
    )
    print(response)
    
    try:
        report = response["choices"][0]["message"]["content"]
        return "📝 Your AI diagnosis summary report is generated! Please review below. Thank you for trying this tool!", report 
    except Exception as e:
        return "Sorry. There was an error in report generation: " + e, "To be generated"
    

with gr.Blocks() as app:
    with gr.Column():
        gr.Markdown(
        """
        # MedAssist-Liver: an AI-powered Liver Tumor Segmentation Tool
        
        Welcome to explore the power of AI for automated medical image analysis with our user-friendly app! 
        
        This tool is designed to assist in the identification and segmentation of liver and tumor from medical images. By uploading a CT scan image, a pre-trained machine learning model will automatically segment the liver and tumor regions. Segmented tumor's characteristics such as shape, size, and location are then analyzed to produce an AI-generated diagnosis report of the liver cancer.
        
        ⚠️ Important disclaimer: these model outputs should NOT replace the medical diagnosis of healthcare professionals. For your reference, our model was trained on the [HCC-TACE-Seg dataset](https://www.cancerimagingarchive.net/collection/hcc-tace-seg/) and achieved 0.954 dice score for liver segmentation and 0.570 dice score for tumor segmentation. Improving tumor segmentation is still an active area of research! 
        """)
    
    with gr.Row():
        comment = gr.Textbox(label='🤖 Your tool guide:', value="👋 Hi there, I will be helping you use this tool. To get started, upload a CT scan image or select one from examples.")
        
    
    with gr.Row():
        
        with gr.Column(scale=2):
            image_file = gr.File(label="Step 1: Upload a CT image (.nrrd)", file_count='single', file_types=['.nrrd'], type='filepath')
            gr.Examples(examples_path, [image_file])
            btn_upload = gr.Button("Upload")
            
        with gr.Column(scale=2):
            selected_mask = gr.CheckboxGroup(label='Step 2: Select mask to produce', choices=['liver mask', 'tumor mask'], value = ['liver mask'])
            btn_segment = gr.Button("Generate Segmentation")
                
    with gr.Row():
        slider = gr.Slider(1, 100, step=1, label="Image slice: ")
        selected_slice = gr.State(value=1)
        
    with gr.Row():
        myimage = gr.AnnotatedImage(label="Image Viewer", height=1000, width=1000, color_map={"segmented liver": "#0373fc", "segmented tumor": "#eb5334"})
        
    with gr.Row():
        with gr.Column(scale=2):
            btn_download_liver = gr.DownloadButton("Download liver mask", visible=False)
        with gr.Column(scale=2):
            btn_download_tumor = gr.DownloadButton("Download tumor mask", visible=False)
        
    with gr.Row():       
        report = gr.Textbox(label='Step 4. Generate summary report using AI:', value="To be generated. ")
            
    with gr.Row():
        btn_report = gr.Button("Generate summary")  

    btn_upload.click(fn=load_image, 
        inputs=[image_file, slider, selected_slice], 
        outputs=[comment, myimage],
    )
    
    btn_segment.click(fn=segment, 
        inputs=[image_file, selected_mask, slider, selected_slice], 
        outputs=[comment, btn_download_liver, btn_download_tumor, myimage],
    )
    
    slider.change(
        render,
        inputs=[image_file, slider, selected_slice],
        outputs=[myimage]
    )
    
    btn_report.click(fn=generate_summary,
        inputs=[image_file],
        outputs=[comment, report]
    )
    
    
app.launch()




    