import os
from sklearn.model_selection import train_test_split
import monai
from monai.data import Dataset, DataLoader
from data_transforms import define_transforms, define_transforms_loadonly
import torch 
import numpy as np
from visualization import visualize_patient
from monai.data import list_data_collate
import pandas as pd


def prepare_clinical_data(data_file, predictors):
    
    # read data file 
    info = pd.read_excel(data_file, sheet_name=0)
    
    # convert to numerical 
    info['CPS'] = info['CPS'].map({'A': 1, 'B': 2, 'C': 3})
    info['T_involvment'] = info['T_involvment'].map({'< or = 50%': 1, '>50%': 2})
    info['CLIP_Score'] = info['CLIP_Score'].map({'Stage_0': 0, 'Stage_1': 1, 'Stage_2': 2, 'Stage_3': 3, 'Stage_4': 4, 'Stage_5': 5, 'Stage_6': 6})
    info['Okuda'] = info['Okuda'].map({'Stage I': 1, 'Stage II': 2, 'Stage III': 3})
    info['TNM'] = info['TNM'].map({'Stage-I': 1, 'Stage-II': 2, 'Stage-IIIA': 3, 'Stage-IIIB': 4, 'Stage-IIIC': 5, 'Stage-IVA': 6, 'Stage-IVB': 7})
    info['BCLC'] = info['BCLC'].map({'0': 0, 'Stage-A': 1, 'Stage-B': 2, 'Stage-C': 3, 'Stage-D': 4})
    
    # remove duplicates 
    info.groupby("TCIA_ID").first() 
    
    # select columns 
    info = info[['TCIA_ID'] + predictors].rename(columns={'TCIA_ID': "patient_id"})
    
    
    return info
    


def preparare_train_test_txt(data_dir, test_patient_ratio=0.2, seed=1):
    """
    From a list of patients, split them into train and test and export list to .txt files 
    """
    
    # split based on seed, write to txt files
    patients = os.listdir(data_dir)
    patients.remove("HCC-TACE-Seg_clinical_data-V2.xlsx")
    patients = list(set(patients))
    
    # remove one patient with wrong labels
    try:
        patients.remove("HCC_017")
        print("The patient HCC_017 is removed due to label issues including necrosis.")
    except Exception as e:
        pass 
    
    print("Total patients:", len(patients))
    patients_train, patients_test = train_test_split(patients, test_size=test_patient_ratio, random_state=seed)
    print("   There are", len(patients_train), "patients in training")
    print("   There are", len(patients_test), "patients in test")

    # export a copy
    if not os.path.exists('train-test-split-seed' + str(seed)):
        os.makedirs('train-test-split-seed' + str(seed))
    with open(r'train-test-split-seed' + str(seed) + '/train.txt', 'w') as f:
        f.write(','.join(patient for patient in patients_train))
    with open(r'train-test-split-seed' + str(seed) + '/test.txt', 'w') as f:
        f.write(','.join(patient for patient in patients_test))
    
    print("Files saved to", 'train-test-split-seed' + str(seed) + '/train.txt and train-test-split-seed' + str(seed) + '/test.txt')
    return


    

def extract_file_path(patient_id, data_folder):
    """
    Given one patient's ID, obtain the file path of the image and mask data. 
    If patient has multiple images, they are labeled as pre1, pre2, etc. 
    """
    path = os.path.join(data_folder, patient_id)
    files = os.listdir(path)
    patient_files = {}
    count = 1
    for file in files:
      if "seg" in file or "Segmentation" in file:
        patient_files["mask"] = os.path.join(path, file)
      else:
        patient_files["pre_" + str(count)] = os.path.join(path, file)
        count += 1
    return patient_files
    
    
    
def get_patient_dictionaries(txt_file, data_dir):
    """
    From .txt file that stores list of patients, look through data folders and extract a dictionary of patient data 
    """
    assert os.path.isfile(txt_file), "The file " + txt_file + " was not found. Please check your file directory."
        
    file = open(txt_file, "r")
    patients = file.read().split(',')

    data_dict = []

    for patient_id in patients:

      # get directories for mask and images
      patient_files = extract_file_path(patient_id, data_dir)

      # pair up each image with the mask
      for key, value in patient_files.items():
        if key != "mask":
          data_dict.append(
              {
                "patient_id": patient_id,
                "image": patient_files[key],
                "mask": patient_files["mask"]
              }
          )

    print("   There are", len(data_dict), "image-masks in this dataset.")
    return data_dict
    
    


def build_dataset(config, get_clinical=False):

    def custom_collate_fn(batch):
        """
        Custom collate function to stack samples along the first dimension.

        Args:
            batch (list): List of dictionaries with keys "image" and "mask",
                          where values are tensors of shape (N, 1, 512, 512).

        Returns:
            tuple: Tuple containing two tensors:
                  - Stacked images of shape (B, 1, 512, 512)
                  - Stacked masks of shape (B, 1, 512, 512)
                  where B is the total number of samples in the batch.
        """
        # torch.manual_seed(1)
        num_samples_to_select = config['BATCH_SIZE']

        # Extract images and masks from the batch
        images, masks = [], []
        for sample in batch:
            num_samples = min(sample["image"].shape[0], sample["mask"].shape[0])
            random_indices = torch.randperm(num_samples)[:num_samples_to_select]
            if "3D" in config['MODEL_NAME']: # 3D image
                images.append(sample["image"][:,:512,:512,:]) # ensure image and mask same size
                masks.append(sample["mask"][:,:512,:512,:])
            else:
                images.append(sample["image"][random_indices,:,:512,:512]) # ensure image and mask same size
                masks.append(sample["mask"][random_indices,:,:512,:512])
                #images.append(sample["image"][:,:,:512,:512]) # ensure image and mask same size
                #masks.append(sample["mask"][:,:,:512,:512])

        # Stack images and masks along the first dimension
        try:
            if "3D" not in config['MODEL_NAME']: # 3D image
                concatenated_images = torch.cat(images, dim=0)
                concatenated_masks = torch.cat(masks, dim=0)
            else:
                concatenated_images = torch.stack(images, dim=0)
                concatenated_masks = torch.stack(masks, dim=0)
        except Exception as e:
            print("WARNING: not all images/masks are 512 by 512. Please check. ", images[0].shape, images[1].shape, masks[0].shape, masks[1].shape)
            return None, None

        # Return stacked images and masks as tensors
        return {"image": concatenated_images, "mask": concatenated_masks}

    # get list of training and test patient files
    train_data_dict = get_patient_dictionaries(config['TRAIN_PATIENTS_FILE'], config['DATA_DIR'])
    test_data_dict = get_patient_dictionaries(config['TEST_PATIENTS_FILE'], config['DATA_DIR'])
    if config['ONESAMPLETESTRUN']: train_data_dict = train_data_dict[:2]
    ttrain_data_dict, valid_data_dict = train_test_split(train_data_dict, test_size=config['VALID_PATIENT_RATIO'], shuffle=False, random_state=1) # must be false to match with linical data 
    print("   Training patients:", len(ttrain_data_dict), " Validation patients:", len(valid_data_dict))
    print("   Test patients:", len(test_data_dict))

    # define data transformations
    preprocessing_transforms_train, preprocessing_transforms_test, postprocessing_transforms = define_transforms(config)

    # create data loaders
    train_ds = Dataset(ttrain_data_dict, transform=preprocessing_transforms_train)
    valid_ds = Dataset(valid_data_dict, transform=preprocessing_transforms_test)
    test_ds = Dataset(test_data_dict, transform=preprocessing_transforms_test)

    if "3D" in config['MODEL_NAME']:
        train_loader = DataLoader(train_ds, batch_size=config['BATCH_SIZE'], collate_fn=custom_collate_fn, shuffle=False, num_workers=config['NUM_WORKERS']) 
        valid_loader = DataLoader(valid_ds, batch_size=config['BATCH_SIZE'], collate_fn=custom_collate_fn, shuffle=False, num_workers=config['NUM_WORKERS'])
        test_loader = DataLoader(test_ds, batch_size=config['BATCH_SIZE'], collate_fn=custom_collate_fn, shuffle=False, num_workers=config['NUM_WORKERS'])
    else:
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, collate_fn=custom_collate_fn, num_workers=config['NUM_WORKERS']) #, pin_memory=torch.cuda.is_available())
        valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=False, collate_fn=custom_collate_fn, num_workers=config['NUM_WORKERS']) #, pin_memory=torch.cuda.is_available())
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=custom_collate_fn, num_workers=config['NUM_WORKERS']) #, pin_memory=torch.cuda.is_available())

    # get clinical data 
    df_clinical_train = pd.DataFrame()
    if get_clinical: 
        # define transforms 
        simple_transforms = define_transforms_loadonly()
        simple_train_ds = Dataset(train_data_dict, transform=simple_transforms)
        simple_train_loader = DataLoader(simple_train_ds, batch_size=config['BATCH_SIZE'], collate_fn=list_data_collate, shuffle=False, num_workers=config['NUM_WORKERS']) #, pin_memory=torch.cuda.is_available())
        
        # compute tumor ratio within liver 
        df_clinical_train['patient_id'] = [p["patient_id"] for p in train_data_dict] 
        ratios_train, ratios_test = [], []
        for batch_data in simple_train_loader:
            labels = batch_data["mask"]
            ratio = torch.sum(labels == 2, dim=(1, 2, 3, 4)) / torch.sum(labels > 0, dim=(1, 2, 3, 4))
            ratios_train.append(ratio.cpu().numpy()[0]) # [metatensor()]
        df_clinical_train['tumor_ratio'] = ratios_train
        
        # get clinical features 
        info = prepare_clinical_data(config['CLINICAL_DATA_FILE'], config['CLINICAL_PREDICTORS'])
        df_clinical_train = pd.merge(df_clinical_train, info, on='patient_id', how="left")
        df_clinical_train.fillna(df_clinical_train.median(), inplace=True)
        df_clinical_train.set_index("patient_id", inplace=True)
        
    # visualize the data loader for one image to ensure correct formatting
    print("Example data transformations:")
    while True:
        sample = preprocessing_transforms_train(train_data_dict[0])
        if isinstance(sample, list): # depending on preprocessing, one sample may be [sample] or sample 
            sample = sample[0]
        if torch.sum(sample['mask'][-1]) == 0: continue
        print(f"  image shape: {sample['image'].shape}")
        print(f"  mask shape: {sample['mask'].shape}")
        print(f"  mask values: {np.unique(sample['mask'])}")
        #print(f"  image affine:\n{sample['image'].meta['affine']}")
        print(f"  image min max: {np.min(sample['image']), np.max(sample['image'])}")
        visualize_patient(sample['image'], sample['mask'], n_slices=3, z_dim_last="3D" in config['MODEL_NAME'], mask_channel=-1)
        break

    temp = monai.utils.first(test_loader)
    print("Test loader shapes:", temp['image'].shape, temp['mask'].shape)
        
    return train_loader, valid_loader, test_loader, postprocessing_transforms, df_clinical_train
    
    