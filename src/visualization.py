from matplotlib import pyplot as plt 
import math 
import numpy as np



def visualize_results(img, mask, pred, n_slices: int=3, slices: list=None, title: str=""):
    """
    img: tensor [C, H, W, Z]
    mask: tensor [C, H, W, Z]
    pred: tensor [C, H, W, Z]
    n_slices: number of slices to visualize
    slices: list of slices to visualize
    title; title of the plot 
    """
    if slices is not None:
      n_slices = len(slices)

    fig, ax = plt.subplots(n_slices, 3, figsize=(14, 5*n_slices))
    inc = img.shape[-1] // n_slices
    mask_masked = np.ma.masked_where(mask == 0, mask)
    pred_masked = np.ma.masked_where(pred == 0, pred)

    for i in range(n_slices):
        slice_num = i*inc if slices is None else slices[i]
        
        # image 
        for c in range(3):
          ax[i,c].imshow(img[0,:,:,slice_num], cmap="gray")
          ax[i,c].axis("off")
          ax[i,c].set_title(f'image')

        # ground truth 
        ax[i,1].imshow(mask_masked[1,:,:,slice_num], cmap='jet', vmin=1, vmax=4, interpolation='none', alpha=0.5)
        ax[i,1].imshow(mask_masked[2,:,:,slice_num], cmap='Reds', vmin=0, vmax=1.3, interpolation='none', alpha=0.8)
        ax[i,1].set_title(f'ground truth')
        
        # predicted 
        ax[i,2].imshow(pred_masked[1,:,:,slice_num], cmap='jet', vmin=1, vmax=4, interpolation='none', alpha=0.5)
        ax[i,2].imshow(pred_masked[2,:,:,slice_num], cmap='Reds', vmin=0, vmax=1.3, interpolation='none', alpha=0.8)
        ax[i,2].set_title(f'predicted')

    plt.suptitle(title, size=14)
    plt.tight_layout()
    plt.show()
    

def visualize_patient(img, mask=None, n_slices: int=3, slices: list=None, z_dim_last=True, mask_channel=0, title: str=""):
    """
    img: tensor [C, H, W, Z]
    mask: tensor [C, H, W, Z]
    n: number of slices to visualize
    """
    if slices is not None:
      n_slices = len(slices)

    fig, ax = plt.subplots(math.ceil(n_slices/3), 3, figsize=(14, 5*math.ceil(n_slices/3)))
    if z_dim_last: inc = img.shape[-1] // n_slices
    else: inc = img.shape[0] // n_slices
    masked = np.ma.masked_where(mask == 0, mask)

    for i in range(n_slices):
        r, c = divmod(i, 3)
        slice_num = i*inc if slices is None else slices[i]
        if n_slices <= 3:
            if z_dim_last: ax[c].imshow(img[0,:,:,slice_num], cmap="gray")
            else: ax[c].imshow(img[slice_num,0,:,:], cmap="gray")
            ax[c].axis("off")
            ax[c].set_title(f'slice {slice_num}')
            if mask is not None:
                if z_dim_last: mask_overlay = ax[c].imshow(masked[mask_channel,:,:,slice_num], cmap='jet', vmin=1, vmax=4, interpolation='none', alpha=0.4)
                else: mask_overlay = ax[c].imshow(masked[slice_num,mask_channel,:,:], cmap='jet', vmin=1, vmax=4, interpolation='none', alpha=0.4)
        else:
            if z_dim_last: ax[r][c].imshow(img[0,:,:,slice_num], cmap="gray")
            else: ax[r][c].imshow(img[slice_num,0,:,:], cmap="gray")
            ax[r][c].axis("off")
            ax[r][c].set_title(f'slice {slice_num}')
            if mask is not None:
                if z_dim_last: mask_overlay = ax[r][c].imshow(masked[mask_channel,:,:,slice_num], cmap='jet', vmin=1, vmax=4, interpolation='none', alpha=0.4)
                else: mask_overlay = ax[r][c].imshow(masked[slice_num,mask_channel,:,:], cmap='jet', vmin=1, vmax=4, interpolation='none', alpha=0.4)

    plt.suptitle(title, size=14)
    #if mask is not None:
    #  cbar = fig.colorbar(mask_overlay, extend='both')
    plt.tight_layout()
    plt.show()
    
    fig, ax = plt.subplots(math.ceil(n_slices/3), 3, figsize=(14, 5*math.ceil(n_slices/3)))
    if z_dim_last: inc = img.shape[-1] // n_slices
    else: inc = img.shape[0] // n_slices

    for i in range(n_slices):
        r, c = divmod(i, 3)
        slice_num = i*inc if slices is None else slices[i]
        if n_slices <= 3:
            if z_dim_last: ax[c].imshow(img[0,:,:,slice_num], cmap="gray")
            else: ax[c].imshow(img[slice_num,0,:,:], cmap="gray")
            ax[c].axis("off")
            ax[c].set_title(f'slice {slice_num}')
        else:
            if z_dim_last: ax[r][c].imshow(img[0,:,:,slice_num], cmap="gray")
            else: ax[r][c].imshow(img[slice_num,0,:,:], cmap="gray")
            ax[r][c].axis("off")
            ax[r][c].set_title(f'slice {slice_num}')

    plt.suptitle(title, size=14)

    plt.tight_layout()
    plt.show()