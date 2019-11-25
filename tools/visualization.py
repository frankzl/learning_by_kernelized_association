import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math

def imshow(images, labels = [], num_row=5, wspace=None, hspace=1, figsize=(10,10), imgwidth=28, channels=1, cmap=False):
    
    labels = len(images)*[" "] if (len(labels) == 0) else labels
    
    gs = gridspec.GridSpec(math.ceil(len(images)/num_row),num_row, hspace=hspace, wspace=wspace)
    f  = plt.figure(figsize=figsize)
    
    for idx in range(len(images)):
        ax = plt.subplot(gs[idx])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title(labels[idx])
        if channels == 1:
            im = ax.imshow(images[idx].reshape(-1,imgwidth))
        else: im = ax.imshow(images[idx].reshape(-1,imgwidth, channels))
        
        if cmap:
            f.colorbar(im, ax=ax)


    
