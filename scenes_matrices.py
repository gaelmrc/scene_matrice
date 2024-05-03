import utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image

def scene_matrice(nb_clusters,image_path, patch_size, overlap_percent=0.5):
    patches, coord, width, height = utils.crop_image_to_patches(image_path,patch_size,overlap_percent)
    texts = utils.patch2text(patches)
    matsim = utils.matsim(texts)
    mat_clusters = utils.pixel_level_clusters(matsim, nb_clusters, coord, width, height, patch_size)

    return texts, matsim, mat_clusters

def display(auth_img_path, nb_clusters, mask_clusters, overlap, patch_size):
    auth_img = Image.open(auth_img_path)
    width,height = auth_img.size
    colors = plt.cm.viridis(np.linspace(0, 1, nb_clusters)) #palette de couleurs
    mask_image = np.zeros((mask_clusters.shape[0], mask_clusters.shape[1],4))
    for i in range(len(colors)):
        mask_image[mask_clusters==i] = colors[i]
    mask_image = Image.fromarray((mask_image * 255).astype(np.uint8))
    superposed_image = Image.blend(auth_img.convert("RGBA"), mask_image, alpha=0.5)
    
    fig, ax = plt.subplots()
    ax.imshow(superposed_image)
    ax.axis('off')

    hyperparams = {'Overlap': f'{overlap}',
                    'Patchsize': f'{patch_size}',
                    'Number of clusters': f'{nb_clusters}',
                    'Image size': f'{width}*{height}'}
    legend_text = '\n'.join([f'{key}: {value}' for key, value in hyperparams.items()])
    fig.subplots_adjust(left = 0.15, right = 0.85, top = 0.9,bottom = 0.15)
    plt.figtext(0.5,0.01, legend_text, ha="center", fontsize=11, bbox={"facecolor":"orange", "alpha":0.5, "pad":3})
    plt.savefig('exp/'+f'{overlap}_{patch_size}_{nb_clusters}.png')
