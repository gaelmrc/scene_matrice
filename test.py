import scenes_matrices
import numpy as np 
import matplotlib.pyplot as plt
import time
import os

def experiment(image_path = '/home/aurelien/Bureau/scene_matrices/image059.png'):
    overlap_start = 0.50
    patch_size_start = 150
    for overlap in np.arange(overlap_start, 0.8, 0.10):
        for patchsize in range(patch_size_start, 525, 75):
            for clusters in range(2, 5):
                if f'{overlap}_{patchsize}_{clusters}.png' not in os.listdir('exp/'):
                    start = time.time()
                    texts, matsim, mat_clusters = scenes_matrices.scene_matrice(clusters, image_path, patch_size = patchsize, overlap_percent = overlap)
                    scenes_matrices.display(auth_img_path = image_path, nb_clusters=clusters, mask_clusters = np.transpose(mat_clusters), overlap = overlap, patch_size= patchsize)
                    print(time.time()-start)
                
experiment()