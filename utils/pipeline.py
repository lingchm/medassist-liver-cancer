import logging
import sys
import tempfile
from glob import glob
from torchsummary import summary
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
import torchvision
import monai
from monai.metrics import DiceMetric, ConfusionMatrixMetric, MeanIoU
from monai.visualize import plot_2d_or_3d_image
from visualization import visualize_patient 
from sliding_window import sw_inference
from data_preparation import build_dataset 
from models import UNet2D, UNet3D
from loss import WeaklyDiceFocalLoss
from sklearn.linear_model import LinearRegression
from nrrd import write, read 
import morphsnakes as ms
from monai.data import decollate_batch

    
def build_optimizer(model, config):

    if config['LOSS'] == "gdice":
        loss_function = monai.losses.GeneralizedDiceLoss(
            include_background=config['EVAL_INCLUDE_BACKGROUND'],
            reduction="mean", to_onehot_y=True, sigmoid=True) if len(config['KEEP_CLASSES'])<=2 else monai.losses.GeneralizedDiceLoss(
            include_background=config['EVAL_INCLUDE_BACKGROUND'], reduction="mean", to_onehot_y=False, softmax=True)
    elif config['LOSS'] == 'cdice':
        loss_function = monai.losses.DiceCELoss(
            include_background=config['EVAL_INCLUDE_BACKGROUND'],
            reduction="mean", to_onehot_y=True, sigmoid=True) if len(config['KEEP_CLASSES'])<=2 else monai.losses.DiceCELoss(
            include_background=config['EVAL_INCLUDE_BACKGROUND'], reduction="mean", to_onehot_y=False, softmax=True)
    elif config['LOSS'] == 'mdice':
        loss_function = monai.losses.MaskedDiceLoss()
    elif config['LOSS'] == 'wdice':
        # Example with 3 classes (including the background: label 0).
        # The distance between the background class (label 0) and the other classes is the maximum, equal to 1.
        # The distance between class 1 and class 2 is 0.5.
        dist_mat = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 0.5], [1.0, 0.5, 0.0]], dtype=np.float32)
        loss_function = monai.losses.GeneralizedWassersteinDiceLoss(dist_matrix=dist_mat)
    elif config['LOSS'] == "fdice":
        loss_function = monai.losses.DiceFocalLoss(
            include_background=config['EVAL_INCLUDE_BACKGROUND'], to_onehot_y=True, sigmoid=True) if len(config['KEEP_CLASSES'])<=2 else monai.losses.DiceFocalLoss(
            include_background=config['EVAL_INCLUDE_BACKGROUND'], to_onehot_y=False, softmax=True)
    elif config['LOSS'] == "wfdice":
        loss_function = WeaklyDiceFocalLoss(include_background=config['EVAL_INCLUDE_BACKGROUND'], to_onehot_y=True, sigmoid=True, lambda_weak=config['LAMBDA_WEAK']) if len(config['KEEP_CLASSES'])<=2 else WeaklyDiceFocalLoss(include_background=config['EVAL_INCLUDE_BACKGROUND'], to_onehot_y=False, softmax=True, lambda_weak=config['LAMBDA_WEAK'])
    else:
        loss_function = monai.losses.DiceLoss(
            include_background=config['EVAL_INCLUDE_BACKGROUND'],
            reduction="mean", to_onehot_y=True, sigmoid=True, squared_pred=True) if len(config['KEEP_CLASSES'])<=2 else monai.losses.DiceLoss(
            include_background=config['EVAL_INCLUDE_BACKGROUND'], reduction="mean", to_onehot_y=False, softmax=True, squared_pred=True)

    eval_metrics = [
        ("sensitivity", ConfusionMatrixMetric(include_background=config['EVAL_INCLUDE_BACKGROUND'], metric_name='sensitivity', reduction="mean_batch")),
        ("specificity", ConfusionMatrixMetric(include_background=config['EVAL_INCLUDE_BACKGROUND'], metric_name='specificity', reduction="mean_batch")),
        ("accuracy", ConfusionMatrixMetric(include_background=config['EVAL_INCLUDE_BACKGROUND'], metric_name='accuracy', reduction="mean_batch")),
        ("dice", DiceMetric(include_background=config['EVAL_INCLUDE_BACKGROUND'], reduction="mean_batch")),
        ("IoU", MeanIoU(include_background=config['EVAL_INCLUDE_BACKGROUND'], reduction="mean_batch"))
    ]

    optimizer = torch.optim.Adam(model.parameters(), config['LEARNING_RATE'], weight_decay=1e-5, amsgrad=True)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['MAX_EPOCHS'])
    return loss_function, optimizer, lr_scheduler, eval_metrics


    
def load_weights(model, config):
    try:
        model.load_state_dict(torch.load("checkpoints/" + config['PRETRAINED_WEIGHTS'] + ".pth", map_location=torch.device(config['DEVICE'])))
        print("Model weights from", config['PRETRAINED_WEIGHTS'], "have been loaded")
    except Exception as e:
        try:
            model.load_state_dict(torch.load(config['PRETRAINED_WEIGHTS'], map_location=torch.device(config['DEVICE'])))
            print("Model weights from", config['PRETRAINED_WEIGHTS'], "have been loaded")
        except Exception as e: # load 
            print("WARNING: weights were not loaded. ", e)
            pass    
    
    return model 


def build_model(config):
    
    config = get_defaults(config)
    
    dropout_prob = config['DROPOUT']
    
    if "SegResNetVAE" in config["MODEL_NAME"]:
        model = monai.networks.nets.SegResNetVAE(
            input_image_size=config['ROI_SIZE'] if "3D" in config['MODEL_NAME'] else (config['ROI_SIZE'][0], config['ROI_SIZE'][1]),
            vae_estimate_std=False, 
            vae_default_std=0.3, 
            vae_nz=256, 
            spatial_dims=3 if "3D" in config["MODEL_NAME"] else 2,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            in_channels=1,
            norm='instance',
            out_channels=len(config['KEEP_CLASSES']),
            dropout_prob=dropout_prob,
        ).to(config['DEVICE'])
    
    elif "SegResNet" in config["MODEL_NAME"]:
        model = monai.networks.nets.SegResNet(
            spatial_dims=3 if "3D" in config["MODEL_NAME"] else 2,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            in_channels=1,
            out_channels=len(config['KEEP_CLASSES']),
            dropout_prob=dropout_prob,
            norm="instance"
        ).to(config['DEVICE'])
    
    elif "SwinUNETR" in config["MODEL_NAME"]:
        model = monai.networks.nets.SwinUNETR(
          img_size=config['ROI_SIZE'],
          in_channels=1,
          out_channels=len(config['KEEP_CLASSES']),
          feature_size=48,
          drop_rate=dropout_prob,
          attn_drop_rate=0.0,
          dropout_path_rate=0.0,
          use_checkpoint=True
       ).to(config['DEVICE'])
    
    elif "UNETR" in config["MODEL_NAME"]:
        model = monai.networks.nets.UNETR(
          img_size=config['ROI_SIZE'] if "3D" in config['MODEL_NAME'] else (config['ROI_SIZE'][0], config['ROI_SIZE'][1]),
          in_channels=1,
          out_channels=len(config['KEEP_CLASSES']),
          feature_size=16,
          hidden_size=256,
          mlp_dim=3072,
          num_heads=8,
          pos_embed="perceptron",
          norm_name="instance",
          res_block=True,
          dropout_rate=dropout_prob,
      ).to(config['DEVICE'])
    
    elif "MANet" in config["MODEL_NAME"]:
        if "2D" in config["MODEL_NAME"]:
            model = UNet2D(
                1, 
                len(config['KEEP_CLASSES']), 
                pab_channels=64, 
                use_batchnorm=True
                ).to(config['DEVICE'])
        else:
            model = UNet3D(
                1, 
                len(config['KEEP_CLASSES']), 
                pab_channels=32, 
                use_batchnorm=True
                ).to(config['DEVICE'])
    
    elif "UNetPlusPlus" in config["MODEL_NAME"]:
        model = monai.networks.nets.BasicUNetPlusPlus(
            spatial_dims=3 if "3D" in config["MODEL_NAME"] else 2,
            in_channels=1,
            out_channels=len(config['KEEP_CLASSES']),
            features=(32, 32, 64, 128, 256, 32),
            norm="instance",
            dropout=dropout_prob,
        ).to(config['DEVICE'])
    
    elif "UNet1" in config['MODEL_NAME']:
        model = monai.networks.nets.UNet(
            spatial_dims=3 if "3D" in config["MODEL_NAME"] else 2,
            in_channels=1,
            out_channels=len(config['KEEP_CLASSES']),
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm="instance"
        ).to(config['DEVICE'])
    
    elif "UNet2" in config['MODEL_NAME']:
        model = monai.networks.nets.UNet(
            spatial_dims=3 if "3D" in config["MODEL_NAME"] else 2,
            in_channels=1,
            out_channels=len(config['KEEP_CLASSES']),
            channels=(32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=4,
            norm="instance"
        ).to(config['DEVICE'])
        
    else:
        print(config["MODEL_NAME"], "is not a valid model name")
        return None

    try:
        if "3D" in config['MODEL_NAME']:
            print(summary(model, input_size=(1, config['ROI_SIZE'][0], config['ROI_SIZE'][1], config['ROI_SIZE'][2])))
        else:
            print(summary(model, input_size=(1, config['ROI_SIZE'][0], config['ROI_SIZE'][1])))
    except Exception as e:
        print("could not load model summary:", e)

    if config['PRETRAINED_WEIGHTS'] is not None and config['PRETRAINED_WEIGHTS']:
        model = load_weights(model, config)
    return model


def train(model, train_loader, val_loader, loss_function, eval_metrics, optimizer, config,
          scheduler=None, writer=None, postprocessing_transforms = None, weak_labels = None):

    if writer is None: writer = SummaryWriter(log_dir="runs/" + config['EXPORT_FILE_NAME'])
    best_metric, best_metric_epoch = -1, -1
    prev_metric, patience, patience_counter = 1, config['EARLY_STOPPING_PATIENCE'], 0
    if config['AUTOCAST']: scaler = GradScaler() # Initialize GradScaler for mixed precision training

    for epoch in range(config['MAX_EPOCHS']):
        print("-" * 10)
        model.train()
        epoch_loss, step = 0, 0
        with tqdm(train_loader) as progress_bar:
            for batch_data in progress_bar:
                step += 1
                inputs, labels = batch_data["image"].to(config['DEVICE']), batch_data["mask"].to(config['DEVICE'])
                
                # only train with batches that have tumor; skip those without tumor  
                if config['TYPE'] == "tumor":
                    if torch.sum(labels[:,-1]) == 0:
                        continue
                    
                # check input shapes 
                if inputs is None or labels is None:
                    continue
                if inputs.shape[-1] != labels.shape[-1] or inputs.shape[0] != labels.shape[0]:
                    print("WARNING: Batch skipped. Image and mask shape does not match:", inputs.shape[0], labels.shape[0])
                    continue

                optimizer.zero_grad()
                if not config['AUTOCAST']:
                    
                    # segmentation output 
                    outputs = model(inputs)
                    if "SegResNetVAE" in config["MODEL_NAME"]: outputs = outputs[0]
                    if isinstance(outputs, list): outputs = outputs[0]

                    # loss 
                    if weak_labels is not None: 
                        weak_label = torch.tensor([weak_labels[step]]).to(config['DEVICE'])
                    loss = loss_function(outputs, labels, weak_label) if config['LOSS'] == 'wfdice' else loss_function(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                else:
                    with autocast():
                        outputs = model(inputs)
                        if "SegResNetVAE" in config["MODEL_NAME"]: outputs = outputs[0]
                        if isinstance(outputs, list): outputs = outputs[0]
                        loss = loss_function(outputs, labels, [weak_labels[step]]) if config['LOSS'] == 'wfdice' else loss_function(outputs, labels)
                        
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if torch.isinf(loss).any():
                        print("Detected inf in gradients.")
                    else:
                        scaler.step(optimizer)
                        scaler.update()

                epoch_loss += loss.item()
                progress_bar.set_description(f'Epoch [{epoch+1}/{config["MAX_EPOCHS"]}], Loss: {epoch_loss/step:.4f}')

        epoch_loss /= step
        writer.add_scalar("train_loss_epoch", epoch_loss, epoch)
        progress_bar.set_description(f'Epoch [{epoch+1}/{config["MAX_EPOCHS"]}], Loss: {epoch_loss:.4f}')

        # validation
        if (epoch + 1) % config['VAL_INTERVAL'] == 0:
            
            # get a list of validation measures, pick one to be the decision maker 
            val_metrics, (val_images, val_labels, val_outputs) = evaluate(model, val_loader, eval_metrics, config, postprocessing_transforms)
            if isinstance(config['EVAL_METRIC'], list):
                cur_metric = np.mean([val_metrics[m] for m in config['EVAL_METRIC']])
            else:
                cur_metric = val_metrics[config['EVAL_METRIC']]
            
            # determine if better than previous best validation metric 
            if cur_metric > best_metric:
                best_metric, best_metric_epoch = cur_metric, epoch + 1
                torch.save(model.state_dict(), "checkpoints/" + config['EXPORT_FILE_NAME'] + ".pth")
            
            # early stopping
            patience_counter = patience_counter + 1 if prev_metric > cur_metric else 0
            if patience_counter == patience or epoch - best_metric_epoch > patience:
                print("Early stopping at epoch", epoch + 1)
                break
            print(f'Current epoch: {epoch + 1} current avg {config["EVAL_METRIC"]}: {cur_metric :.4f} best avg {config["EVAL_METRIC"]}: {best_metric:.4f} at epoch {best_metric_epoch}')
            prev_metric = cur_metric
            
            # writer
            for key, value in val_metrics.items():
                writer.add_scalar("val_" + key, value, epoch)
            plot_2d_or_3d_image(val_images, epoch + 1, writer, index=len(val_outputs)//2, tag="image",frame_dim=-1)
            plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=len(val_outputs)//2, tag="label",frame_dim=-1)
            plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=len(val_outputs)//2, tag="output",frame_dim=-1)

        # update scheduler
        try:
            if scheduler is not None: scheduler.step()
        except:
            pass 

    print(f"Train completed, best {config['EVAL_METRIC']}: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()
    return model, writer



def evaluate(model, val_loader, eval_metrics, config, postprocessing_transforms=None, use_liver_seg=False, export_filenames = [], export_file_metadata = []):

  val_metrics = {}
  model.eval()
  with torch.no_grad():

    step = 0
    for val_data in val_loader:
        # 3D: val_images has shape (1,C,H,W,Z)
        # 2D: val_images has shape (B,C,H,W)
        val_images, val_labels = val_data["image"].to(config['DEVICE']), val_data["mask"].to(config['DEVICE'])
        if use_liver_seg: val_liver = val_data["pred_liver"].to(config['DEVICE'])
                
        if (val_images[0].shape[-1] != val_labels[0].shape[-1]) or (
            "3D" not in config["MODEL_NAME"] and val_images.shape[0] != val_labels.shape[0]):
                print("WARNING: Batch skipped. Image and mask shape does not match:", val_images.shape, val_labels.shape)
                continue

        # convert outputs to probability
        if "3D" in config["MODEL_NAME"]:
            val_outputs = sw_inference(model, val_images, config['ROI_SIZE'], config['AUTOCAST'], discard_second_output='SegResNetVAE' in config['MODEL_NAME'])
        else:
            if "SegResNetVAE" in config["MODEL_NAME"]: val_outputs, _ = model(val_images)
            else: val_outputs = model(val_images)

        # post-procesing
        if postprocessing_transforms is not None:
            val_outputs = [postprocessing_transforms(i) for i in decollate_batch(val_outputs)]

        # remove tumor predictions outside liver 
        for i in range(len(val_outputs)):
            val_outputs[i][-1][torch.where(val_images[i][0] <= 1e-6)] = 0
  
        # apply morphological snakes algorithm 
        if config['POSTPROCESSING_MORF']: 
            for i in range(len(val_outputs)):
                val_outputs[i][-1] = torch.from_numpy(ms.morphological_chan_vese(val_images[i][0].cpu(), iterations=2, init_level_set=val_outputs[i][-1].cpu())).to(config['DEVICE'])
  
        for i in range(len(val_outputs)):
            if use_liver_seg: 
                # use liver model outputs for liver channel 
                val_outputs[i][1] = val_liver[i]
                # if region is tumor, assign liver prediction to 0 
                val_outputs[i][1] -= val_outputs[i][2]
        
        # compute metric for current iteration
        for metric_name, metric in eval_metrics:
            if isinstance(val_outputs[0], list):
                val_outputs = val_outputs[0]
            metric(val_outputs, val_labels)
        
        # save prediction to local folder
        if len(export_filenames) > 0:
            for _ in range(len(val_outputs)):
                numpy_array = val_outputs[_].cpu().detach().numpy()
                write(export_filenames[step], numpy_array[-1], header=export_file_metadata[step])
                print("   Segmentation exported to", export_filenames[step]) 
                step += 1
            
    # aggregate the final mean metric
    for metric_name, metric in eval_metrics:
        if "dice" in metric_name or "IoU" in metric_name: metric_value = metric.aggregate().tolist() 
        else: metric_value = metric.aggregate()[0].tolist() # a list of accuracies, one per class
        val_metrics[metric_name + "_avg"] = np.mean(metric_value)
        if config['TYPE'] != "liver":  
            for c in range(1, len(metric_value) + 1): # class-wise accuracies
                val_metrics[metric_name + "_class" + str(c)] = metric_value[c-1]
        metric.reset()

    return val_metrics, (val_images, val_labels, val_outputs)
    
    
    

def get_defaults(config):
    
    if 'TRAIN' not in config.keys(): config['TRAIN'] = True
    if 'VALID_PATIENT_RATIO' not in config.keys(): config['VALID_PATIENT_RATIO'] = 0.2
    if 'VAL_INTERVAL' not in config.keys(): config['VAL_INTERVAL'] = 1
    if 'VAL_INTERVAL' not in config.keys(): config['DROPOUT'] = 0.1
    if 'EARLY_STOPPING_PATIENCE' not in config.keys(): config['EARLY_STOPPING_PATIENCE'] = 20
    if 'AUTOCAST' not in config.keys(): config['AUTOCAST'] = False
    if 'NUM_WORKERS' not in config.keys(): config['NUM_WORKERS'] = 0
    if 'DROPOUT' not in config.keys(): config['DROPOUT'] = 0.1
    if 'ONESAMPLETESTRUN' not in config.keys(): config['ONESAMPLETESTRUN'] = False
    if 'TRAIN' not in config.keys(): config['TRAIN'] = True
    if 'DATA_AUGMENTATION' not in config.keys(): config['DATA_AUGMENTATION'] = False
    if 'POSTPROCESSING_MORF' not in config.keys(): config['POSTPROCESSING_MORF'] = False
    if 'PREPROCESSING' not in config.keys(): config['PREPROCESSING'] = ""
    if 'PRETRAINED_WEIGHTS' not in config.keys(): config['PRETRAINED_WEIGHTS'] = ""
    
    if 'EVAL_INCLUDE_BACKGROUND' not in config.keys(): 
        if config['TYPE'] == "liver": config['EVAL_INCLUDE_BACKGROUND'] = True
        else: config['EVAL_INCLUDE_BACKGROUND'] = False
    if 'EVAL_METRIC' not in config.keys(): 
        if config['TYPE'] == "liver": config['EVAL_METRIC'] = ["dice_avg"]
        else: config['EVAL_METRIC'] = ["dice_class2"]

    if 'CLINICAL_DATA_FILE' not in config.keys(): config['CLINICAL_DATA_FILE'] = "Dataset/HCC-TACE-Seg_clinical_data-V2.xlsx"
    if 'CLINICAL_PREDICTORS' not in config.keys(): config['CLINICAL_PREDICTORS'] = ['T_involvment', 'CLIP_Score','Personal history of cancer', 'TNM', 'Metastasis','fhx_can', 'Alcohol', 'Smoking', 'Evidence_of_cirh', 'AFP', 'age', 'Diabetes', 'Lymphnodes', 'Interval_BL', 'TTP']
    if 'LAMBDA_WEAK' not in config.keys(): config['LAMBDA_WEAK'] = 0.5
    if 'MASKNONLIVER' not in config.keys(): config['MASKNONLIVER'] = False
    
    if config['TYPE'] == "liver": config['KEEP_CLASSES']=["normal", "liver"]
    elif config['TYPE'] == "tumor": config['KEEP_CLASSES']=["normal", "liver", "tumor"]
    else: config['KEEP_CLASSES'] = ["normal", "liver", "tumor", "portal vein", "abdominal aorta"]
    
    config['DEVICE'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['EXPORT_FILE_NAME'] = config['TYPE']+ "_" + config['MODEL_NAME'] + "_" + config['LOSS'] + "_batchsize" + str(config['BATCH_SIZE']) + "_DA" + str(config['DATA_AUGMENTATION']) + "_HU" + str(config['HU_RANGE'][0]) + "-" + str(config['HU_RANGE'][1]) + "_" + config['PREPROCESSING'] + "_" + str(config['ROI_SIZE'][0]) + "_" + str(config['ROI_SIZE'][1]) + "_" + str(config['ROI_SIZE'][2]) + "_dropout" + str(config['DROPOUT'])
    if config['MASKNONLIVER']: config['EXPORT_FILE_NAME'] += "_wobackground"
    if config['LOSS'] == "wfdice": config['EXPORT_FILE_NAME'] += "_weaklambda" + str(config['LAMBDA_WEAK'])
    if config['PRETRAINED_WEIGHTS'] != "" and config['PRETRAINED_WEIGHTS'] != config['EXPORT_FILE_NAME']: config['EXPORT_FILE_NAME'] += "_pretraining"
    if config['POSTPROCESSING_MORF']: config['EXPORT_FILE_NAME'] += "_wpostmorf"
    if not config['EVAL_INCLUDE_BACKGROUND']: config['EXPORT_FILE_NAME'] += "_evalnobackground"

    return config 
    

def train_clinical(df_clinical):

    clinical_model = LinearRegression()
    
    # train model 
    print("Training model using", df_clinical.loc[:, df_clinical.columns != 'tumor_ratio'].shape[1], "features")
    print(df_clinical.head())
    clinical_model.fit(df_clinical.loc[:, df_clinical.columns != 'tumor_ratio'], df_clinical['tumor_ratio'])
    
    # obtain predicted ratios 
    pred = clinical_model.predict(df_clinical.loc[:, df_clinical.columns != 'tumor_ratio'])
    
    # evaluate
    corr = np.corrcoef(pred, df_clinical['tumor_ratio'])[0][1]
    mae = np.mean(np.abs(pred - df_clinical['tumor_ratio']))
    print(f"The clinical model was fitted. Corr = {corr: .6f}  MAE = {mae: .6f}")
    
    return pred 
    
    
def model_pipeline(config=None, plot=True):

    torch.cuda.empty_cache()
    config = get_defaults(config)
    print(f"You Are Running on a: {config['DEVICE']}")
    print("file name:", config['EXPORT_FILE_NAME'])

    writer = SummaryWriter(log_dir="runs/" + config['EXPORT_FILE_NAME'])
    
    # prepare data
    train_loader, valid_loader, test_loader, postprocessing_transforms, df_clinical_train = build_dataset(config, get_clinical=config['LOSS']=="wfdice")
    
    # train clinical model
    if config['LOSS'] == "wfdice": weak_labels = train_clinical(df_clinical_train)
    else: weak_labels = None
    
    # train segmentation model 
    model = build_model(config)
    loss_function, optimizer, lr_scheduler, eval_metrics = build_optimizer(model, config)
    if config['TRAIN']:
        train(model, train_loader, valid_loader, loss_function, eval_metrics, optimizer, config, lr_scheduler, writer, postprocessing_transforms, weak_labels)
        model.load_state_dict(torch.load("checkpoints/" + config['EXPORT_FILE_NAME'] + ".pth", map_location=torch.device(config['DEVICE'])))
    if config['ONESAMPLETESTRUN']:
        return None, None, None 
    
    # test segmentation model 
    test_metrics, (test_images, test_labels, test_outputs) = evaluate(model, test_loader, eval_metrics, config, postprocessing_transforms)
    print("Test metrics")
    for key, value in test_metrics.items():
      print(f"   {key}: {value:.4f}")

    # visualize
    if plot:
        if "3D" in config['MODEL_NAME']:
          visualize_patient(test_images[0].cpu(), mask=test_labels[0].cpu(), n_slices=9, title="ground truth", z_dim_last="3D" in config['MODEL_NAME'], mask_channel=-1)
          visualize_patient(test_images[0].cpu(), mask=test_outputs[0].cpu(), n_slices=9, title="predicted", z_dim_last="3D" in config['MODEL_NAME'], mask_channel=-1)
        else:
          visualize_patient(test_images.cpu(), mask=test_labels.cpu(), n_slices=9, title="ground truth", z_dim_last="3D" in config['MODEL_NAME'], mask_channel=-1)
          visualize_patient(test_images.cpu(), mask=torch.stack(test_outputs).cpu(), n_slices=9, title="predicted", z_dim_last="3D" in config['MODEL_NAME'], mask_channel=-1)

    return (test_images, test_labels, test_outputs)
