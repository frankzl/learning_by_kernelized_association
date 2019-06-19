import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math

def imshow(images, labels = None, num_row=5, hspace=1, figsize=(10,10), imgwidth=28, channels=1):
    
    labels = len(images)*[" "] if labels == None else labels
    
    gs = gridspec.GridSpec(math.ceil(len(images)/num_row),num_row, hspace=hspace)
    f  = plt.figure(figsize=figsize)
    
    for idx in range(len(images)):
        ax = plt.subplot(gs[idx])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title(labels[idx])
        if channels == 1:
            ax.imshow(images[idx].reshape(-1,imgwidth))
        else: ax.imshow(images[idx].reshape(-1,imgwidth, channels))
    
