import numpy as np
import torch
from monai.transforms import (
    Activations, AsDiscreteD, AsDiscrete, Compose, ToTensorD, 
    GaussianSmoothD, LoadImageD, TransposeD, OrientationD, ScaleIntensityRangeD,
    ToTensor, FillHoles, KeepLargestConnectedComponent, NormalizeIntensityD
)
from nrrd import read 
from visualization import visualize_results
from data_preparation import get_patient_dictionaries
from monai.data import Dataset, DataLoader
import os
from data_transforms import ConvertMaskValues, MaskOutNonliver
from pipeline import build_model, evaluate 
    
def run_sequential_inference(txt_file, config_liver, config_tumor, eval_metrics, output_dir, only_tumor=False, export=True):

    def custom_collate_fn(batch):
        num_samples_to_select = config_liver['BATCH_SIZE']

        # Extract images and masks from the batch,  ensure image and mask same size
        images, masks, pred_liver = [], [], []
        for sample in batch:
            num_samples = min(sample["image"].shape[0], sample["mask"].shape[0])
            random_indices = torch.randperm(num_samples)[:num_samples_to_select]
            images.append(sample["image"][:,:512,:512,:]) 
            masks.append(sample["mask"][:,:512,:512,:])

        # Stack images and masks along the first dimension
        try:
            concatenated_images = torch.stack(images, dim=0)
            concatenated_masks = torch.stack(masks, dim=0)
        except Exception as e:
            print("WARNING: not all images/masks are 512 by 512. Please check. ", images[0].shape, images[1].shape, masks[0].shape, masks[1].shape)
            return None, None

        # Return stacked images and masks as tensors
        if "pred_liver" in sample.keys():
            return {"image": concatenated_images, "mask": concatenated_masks, "pred_liver": sample["pred_liver"]}
        else:
            return {"image": concatenated_images, "mask": concatenated_masks}

    ### Model preparation 
    print("")
    print("Loading models....")
    liver_model = build_model(config_liver)
    tumor_model = build_model(config_tumor)

    #### Data preparation 
    print("")
    print("Loading test data....")
    test_data_dict = get_patient_dictionaries(txt_file=txt_file, data_dir=config_liver['DATA_DIR'])
    print("   Number of test patients:", len(test_data_dict))
 
    # assign output file names and paths 
    export_file_metadata = []
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    for patient_dict in test_data_dict:
        patient_folder = os.path.join(output_dir, patient_dict['patient_id'])
        if not os.path.exists(patient_folder): os.makedirs(patient_folder)
        patient_dict['pred_liver'] = os.path.join(patient_folder, "liver_segmentation.nrrd")
        patient_dict['pred_tumor'] = os.path.join(patient_folder, "tumor_segmentation.nrrd")
        export_file_metadata.append(read(patient_dict['image'])[1])
    
    #### Liver segmentation 
    # define liver data loading and preprocessing 
    if not only_tumor:
        print("")
        print("Producing liver segmentations....")
        liver_preprocessing = Compose([
            LoadImageD(keys=["image", "mask"], reader="NrrdReader", ensure_channel_first=True),
            OrientationD(keys=["image", "mask"], axcodes="PLI"),
            ScaleIntensityRangeD(keys=["image"],
                a_min=config_liver['HU_RANGE'][0],
                a_max=config_liver['HU_RANGE'][1],
                b_min=0.0, b_max=1.0, clip=True
            ),
            ConvertMaskValues(keys=["mask"], keep_classes=["liver"]),
            ToTensorD(keys=["image", "mask"])
        ])
    
        liver_postprocessing = Compose([
            Activations(sigmoid=True),
            AsDiscrete(argmax=True, to_onehot=None),
            KeepLargestConnectedComponent(applied_labels=[1]),
            FillHoles(applied_labels=[1]),
            ToTensor()
        ])
        test_ds_liver = Dataset(test_data_dict, transform=liver_preprocessing)
        test_ds_liver = DataLoader(test_ds_liver, batch_size=config_liver['BATCH_SIZE'], collate_fn=custom_collate_fn, shuffle=False, num_workers=config_liver['NUM_WORKERS'])
    
        # produce liver model results 
        test_metrics_liver, sample_output_liver = evaluate(liver_model, test_ds_liver, eval_metrics, config_liver, postprocessing_transforms=liver_postprocessing, export_filenames = [p['pred_liver'] for p in test_data_dict], export_file_metadata=export_file_metadata)

        print("")
        print("==============================")
        print("Liver segmentation test performance ....")
        for key, value in test_metrics_liver.items():
            print(f'   {key.replace("_avg", "_liver")}: {value:.3f}')
        print("==============================")
        
    ##### Tumor segmentation 
    print("")
    print("Producing tumor segmentations....")
    
    # define tumor loading and preprocessing
    tumor_preprocessing = Compose([
        LoadImageD(keys=["image", "mask", "pred_liver"], reader="NrrdReader", ensure_channel_first=True),
        OrientationD(keys=["image", "mask"], axcodes="PLI"),
        MaskOutNonliver(mask_key="pred_liver"), # note that liver's predicted segmentation is used to crop to the liver region 
        ScaleIntensityRangeD(keys=["image"],
            a_min=config_tumor['HU_RANGE'][0],
            a_max=config_tumor['HU_RANGE'][1],
            b_min=0.0, b_max=1.0, clip=True
        ),
        ConvertMaskValues(keys=["mask"], keep_classes=["liver", "tumor"]), # format mask for measuring test performance 
        AsDiscreteD(keys=["mask"], to_onehot=3),           # format mask for measuring test performance 
        ToTensorD(keys=["image", "mask", "pred_liver"])
    ])

    tumor_postprocessing = Compose([
        Activations(sigmoid=True),
        AsDiscrete(argmax=True, to_onehot=3),
        ToTensor()
    ])
 
    test_ds_tumor = Dataset(test_data_dict, transform=tumor_preprocessing)
    test_ds_tumor = DataLoader(test_ds_tumor, batch_size=config_tumor['BATCH_SIZE'], collate_fn=custom_collate_fn, shuffle=False, num_workers=config_tumor['NUM_WORKERS'])

    test_metrics_tumor, sample_output_tumor = evaluate(tumor_model, test_ds_tumor, eval_metrics, config_tumor, tumor_postprocessing, use_liver_seg = True, export_filenames = [p['pred_tumor'] for p in test_data_dict] if export else [], export_file_metadata=export_file_metadata)

    print("")
    print("==============================")
    print("Tumor segmentation test performance ....")
    for key, value in test_metrics_tumor.items():
        if "class2" in key: 
            print(f'   {key.replace("_class2", "_tumor")}: {value:.3f}')
    print("==============================")
    print("")

    #### Visualization 

    # combine liver and tumor segmentations into one segmentation output
    if not only_tumor: sample_output_tumor[2][0][1] = sample_output_liver[2][0][0]

    # visualization 
    print("")
    if not only_tumor:
      visualize_results(sample_output_liver[0][0].cpu(), sample_output_tumor[1][0].cpu(), sample_output_tumor[2][0].cpu(), n_slices=5, title="")
    else:
      visualize_results(sample_output_tumor[0][0].cpu(), sample_output_tumor[1][0].cpu(), sample_output_tumor[2][0].cpu(), n_slices=5, title="")

    return 


