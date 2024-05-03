from PIL import Image
from scipy.spatial.distance import cosine
from scipy.spatial import distance
import numpy as np
import blip
import sentence_transfo
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt

def crop_image_to_patches(image_path, patch_size, overlap_percent):
    image = Image.open(image_path)
    width, height = image.size
    print(width,height)
    # Calculer le pas de déplacement en prenant en compte le pourcentage de chevauchement
    step_size = int(patch_size * (1 - overlap_percent))
    print(step_size)
    patches = []
    coord = []
    # Découper l'image en patchs
    print(height- patch_size+1)
    for y in range(0, height - patch_size + 1, step_size):
        for x in range(0, width - patch_size + 1, step_size):
            box = (x, y, x + patch_size, y + patch_size)
            patch = image.crop(box)
            patches.append(patch)
            coord.append((x,y))
    print('SHAPE COORD', np.shape(coord))
    return patches, coord, width,height

def sentence_similarity(ind_sentence,sentence_list):
    embeddings = sentence_transfo.model.encode(sentence_list)
    n,_ = np.shape(embeddings)
    similarity_list = [1-cosine(embeddings[ind_sentence],embeddings[k]) for k in range(n)]
    return similarity_list

def patch2text(patches):
    texts = []
    for patch in patches:
        inputs = blip.processor(patch, return_tensors="pt").to("cuda")
        out = blip.model.generate(**inputs)
        texts.append(blip.processor.decode(out[0],skip_special_tokens=True))
    return texts

def matsim(texts):
    n = len(texts)
    print(n)
    matsim = []
    for i in range(n):
        column_i = sentence_similarity(i, texts)
        matsim.append(column_i)
    #matsim = np.array(matsim)
    return matsim

def spectral_clustering(nb_clusters, matsim):
    sc = SpectralClustering(nb_clusters, affinity="precomputed", random_state=42)
    labels = sc.fit_predict(matsim)
    return labels

def pixel_level_clusters(matsim, nb_clusters,coord,width,height, patch_size):
    mat_clusters = [[[0 for k in range(nb_clusters)] for j in range(height)] for i in range(width)]
    labels = spectral_clustering(nb_clusters, matsim)
    print('SHAPE MATCLUSTERS', np.shape(mat_clusters))
    for xy in range(len(coord)):
        x,y = coord[xy]
        for i in range (x,x+patch_size):
            for j in range(y, y + patch_size):
                mat_clusters[i][j][labels[xy]]+= patch_size/(distance.euclidean((i,j),(x+patch_size/2,y+patch_size/2))+1)
    mat_clusters = [[np.argmax(mat_clusters[i][j]) for j in range(height)] for i in range(width)]        
    return mat_clusters