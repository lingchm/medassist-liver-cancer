import monai 
import cv2
from monai.transforms import MapTransform
import math 
import numpy as np
import torch
import morphsnakes as ms
import monai
import nrrd
import torchvision.transforms as transforms
from monai.transforms import (
    Activations, AsDiscreteD, AsDiscrete, Compose, CastToTypeD, RandSpatialCropd,
    ToTensorD, CropForegroundD, Resized, GaussianSmoothD, 
    LoadImageD, TransposeD, OrientationD, ScaleIntensityRangeD,
    RandAffineD, ResizeWithPadOrCropd, ToTensor,
    FillHoles, KeepLargestConnectedComponent, HistogramNormalizeD, NormalizeIntensityD
)



def define_transforms_loadonly():
    transformations = Compose([
        LoadImageD(keys=["mask"], reader="NrrdReader", ensure_channel_first=True),
        ConvertMaskValues(keys=["mask"], keep_classes=["liver", "tumor"]),
        ToTensor()
    ])
    return transformations


def define_post_processing(config):
      # Post-processing transforms
      post_processing = [
          # Apply softmax activation to convert logits to probabilities
          Activations(sigmoid=True),
          # Convert predicted probabilities to discrete values (0 or 1)
          AsDiscrete(argmax=True, to_onehot=None if len(config['KEEP_CLASSES']) <= 2 else len(config['KEEP_CLASSES'])),
          # Remove small connected components for 1=liver and 2=tumor
          KeepLargestConnectedComponent(applied_labels=[1]),
          # Fill holes in the binary mask for 1=liver and 2=tumor
          FillHoles(applied_labels=[1]),
          ToTensor()
      ]

      return Compose(post_processing)

def define_transforms(config):

      transformations_test = [
              LoadImageD(keys=["image", "mask"], reader="NrrdReader", ensure_channel_first=True),
              # Orient up and down
              OrientationD(keys=["image", "mask"], axcodes="PLI"),
              ToTensorD(keys=["image", "mask"])
              # histogram equilization or normalization
              # HistogramNormalizeD(keys=["image"], num_bins=256, min=0, max=1),
              # Intensity normalization 
              # NormalizeIntensityD(keys=["image"]),
              #CastToTypeD(keys=["image"], dtype=torch.float32),
              #CastToTypeD(keys=["mask"], dtype=torch.int32),
          ]
    
      if config['MASKNONLIVER']:
          transformations_test.extend(
              [
                MaskOutNonliver(mask_key="mask"),
                CropForegroundD(keys=["image", "mask"], source_key="image", allow_smaller=True), 
              ]
          )
          
      transformations_test.append(
          # Windowing based on liver parameters
          ScaleIntensityRangeD(keys=["image"],
            a_min=config['HU_RANGE'][0],
            a_max=config['HU_RANGE'][1],
            b_min=0.0, b_max=1.0, clip=True
          )
      )
      
      if config['PREPROCESSING'] == "clihe":
          transformations_test.append(CLIHE(keys=["image"]))
      
      elif config['PREPROCESSING'] == "gaussian":
          transformations_test.append(GaussianSmoothD(keys=["image"], sigma=0.5))

      # convert labels to 0,1,2 instead of 0,1,2,3,4
      transformations_test.append(ConvertMaskValues(keys=["mask"], keep_classes=config['KEEP_CLASSES']))

      if len(config['KEEP_CLASSES']) > 2: # NEEDED FOR MULTICLASS  https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_brats21_segmentation_3d.ipynb
          transformations_test.append(AsDiscreteD(keys=["mask"], to_onehot=len(config['KEEP_CLASSES']))) # (N, C, H, W) 2d; (1, C, H, W, Z)
      
      if "3D" not in config['MODEL_NAME']:
          transformations_test.append(TransposeD(keys=["image", "mask"], indices=(3,0,1,2)))

      # training transforms include data augmentation
      transformations_train = transformations_test.copy()
      if config['MASKNONLIVER']: transformations_test = transformations_test[:4] + transformations_test[5:] # do not crop to liver foregroudn 

      if config['DATA_AUGMENTATION']:
          if "3D" in config["MODEL_NAME"]:
              transformations_train.append(
                  RandAffineD(keys=["image", "mask"], prob=0.2, padding_mode="border",
                              mode="bilinear", spatial_size=config['ROI_SIZE'],
                              rotate_range=(0.15,0.15,0.15), #translate_range=(30,30,30), 
                              scale_range=(0.1,0.1,0.1)))
          else:
              transformations_train.append(
                  RandAffineD(keys=["image", "mask"], prob=0.2, padding_mode="border",
                              mode="bilinear", #spatial_size=(512, 512),
                              rotate_range=(0.15,0.15), #translate_range=(30,30), 
                              scale_range=(0.1,0.1)))

      transformations_train.extend(
          [
            RandSpatialCropd(keys=["image", "mask"], roi_size=config['ROI_SIZE'], random_size=False),
            ResizeWithPadOrCropd(keys=["image", "mask"], spatial_size=config['ROI_SIZE'], method="end", mode='constant', value=0)
          ]
      )

      postprocessing_transforms = define_post_processing(config)
      preprocessing_transforms_test = Compose(transformations_test)
      preprocessing_transforms_train = Compose(transformations_train)
      preprocessing_transforms_train.set_random_state(seed=1)
      preprocessing_transforms_test.set_random_state(seed=1)

      return preprocessing_transforms_train, preprocessing_transforms_test, postprocessing_transforms
      
      

class CLIHE(MapTransform):
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(allow_missing_keys)
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if len(data['image'].shape) > 3: # 3D image
                data[key] = self.apply_clahe_3d(data[key]) # [B, 1, H, W, Z]
            else:
                data[key] = self.apply_clahe_2d(data[key]) # [B, 1, H, W, Z]
        return data

    def apply_clahe_3d(self, image):
        image = np.asarray(image)
        clahe_slices = []
        for slice_idx in range(image.shape[-1]):
            # Extract the current slice
            slice_2d = image[0, :, :, slice_idx]

            # Apply CLAHE to the current slice
            # slice_2d = cv2.medianBlur(slice_2d, 5)
            # slice_2d = cv2.anisotropicDiffusion(slice_2d, alpha=0.1, K=1, iterations=50)
            # slice_2d = anisotropic_diffusion(slice_2d)
            # slice_2d = cv2.Sobel(slice_2d, cv2.CV_64F, dx=1, dy=1, ksize=5)
            clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(16,16))
            slice_2d = clahe.apply(slice_2d.astype(np.uint8))
            #cv2.threshold(clahe_slice, 155, 255, cv2.THRESH_BINARY)
            kernel = np.ones((2,2), np.float32)/4
            slice_2d = cv2.filter2D(slice_2d, -1, kernel)
            #t = anisodiff2D(delta_t=0.2,kappa=50)
            #slice_2d = t.fit(slice_2d)

            # Append the CLAHE enhanced slice to the list
            clahe_slices.append(slice_2d)

            # Stack the CLAHE enhanced slices along the slice axis to form the 3D image
            clahe_image = np.stack(clahe_slices, axis=-1)

        return torch.from_numpy(clahe_image[None,:])

    def apply_clahe_2d(self, image):
        image = np.asarray(image)

        clahe = cv2.createCLAHE(clipLimit=5)
        clahe_slice = clahe.apply(image[0].astype(np.uint8))

        return torch.from_numpy(clahe_slice)



class GaussianFilter(MapTransform):
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(allow_missing_keys)
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if len(data['image'].shape) > 3: # 3D image
                data[key] = self.apply_clahe_3d(data[key]) # [B, 1, H, W, Z]
            else:
                data[key] = self.apply_clahe_2d(data[key]) # [B, 1, H, W, Z]
        return data

    def apply_clahe_3d(self, image):
        image = np.asarray(image)
        clahe_slices = []
        for slice_idx in range(image.shape[-1]):
            # Extract the current slice
            slice_2d = image[0, :, :, slice_idx]

            # Apply CLAHE to the current slice
            kernel = np.ones((3,3), np.float32)/9
            slice_2d = cv2.filter2D(slice_2d, -1, kernel)
            
            # Append the CLAHE enhanced slice to the list
            clahe_slices.append(slice_2d)

            # Stack the CLAHE enhanced slices along the slice axis to form the 3D image
            clahe_image = np.stack(clahe_slices, axis=-1)

        return torch.from_numpy(clahe_image[None,:])

    def apply_clahe_2d(self, image):
        image = np.asarray(image)

        kernel = np.ones((3,3), np.float32)/9
        slice_2d = cv2.filter2D(image, -1, kernel)

        return torch.from_numpy(slice_2d)


class Morphsnakes(MapTransform):
    # https://github.com/pmneila/morphsnakes/blob/master/morphsnakes.py
    def __init__(self, allow_missing_keys=False):
        super().__init__(allow_missing_keys)

    def __call__(self, data):
        if np.sum(data['mask'][-1]) > 0:
            res = ms.morphological_chan_vese(data['image'][0], iterations=2, init_level_set=data['mask'][-1])
            data['mask'] = res
        return data
        
        
class MaskOutNonliver(MapTransform):
    def __init__(self, allow_missing_keys=False, mask_key="mask"):
        super().__init__(allow_missing_keys)
        self.mask_key = mask_key

    def __call__(self, data):
        # mask out non-liver regions of an image
        # non-liver regions are liver, tumor, or portal vein
        if data[self.mask_key].shape != data['image'].shape:
            return data
        data['image'][data[self.mask_key] >= 4] = -1000
        data['image'][data[self.mask_key] <= 0] = -1000
        return data
        
        
class ConvertMaskValues(MapTransform):
    def __init__(self, keys, allow_missing_keys=False, keep_classes=["normal", "liver", "tumor"]):
        super().__init__(keys, allow_missing_keys)
        self.keep_classes = keep_classes

    def __call__(self, data):
        # original labels: 0 for normal region, 1 for liver, 2 for tumor mass, 3 for portal vein, and 4 for abdominal aorta.
        # converted labels: 0 for normal region and abdominal aorta, 1 for liver and portal vein, 2 for tumor mass

        for key in self.keys:
          data[key][data[key] > 4] = 4 # one patient had class label = 5, converted to 4
          if key in data:
            if "liver" not in self.keep_classes:
                data[key][data[key] == 1] = 0
            if "tumor" not in self.keep_classes:
                data[key][data[key] == 2] = 1
            if "portal vein" not in self.keep_classes:
                data[key][data[key] == 3] = 1
            if "abdominal aorta" not in self.keep_classes:
                data[key][data[key] >= 4] = 0
        return data
