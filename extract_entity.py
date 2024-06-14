import scenes_matrices
import matplotlib.pyplot as plt
import time
import helpers
import ollama
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas
import helpers.prompts
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stopwords
from gensim.parsing.preprocessing import STOPWORDS as gensim_stopwords
import utils
import uuid
from scipy.signal import savgol_filter

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

def documents2Dataframe(documents) -> pd.DataFrame:
    rows = []
    for chunk in documents:
        row = {
            "text": chunk.page_content,
            **chunk.metadata,
            "chunk_id": uuid.uuid4().hex,
        }
        rows = rows + [row]

    df = pd.DataFrame(rows)
    return df

def normalize_node_names(node_name):
    lemmatizer = WordNetLemmatizer()
    combined_stopwords = set(stopwords.words('english')).union(spacy_stopwords).union(sklearn_stopwords).union(gensim_stopwords)
    node_name = re.sub(r'\W+', ' ', node_name.lower())
    tokens = [lemmatizer.lemmatize(word) for word in node_name.split() if word not in combined_stopwords]
    return ' '.join(tokens)

def read_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"Le fichier {file_path} n'a pas été trouvé.")
        return None
    except IOError as e:
        print(f"Une erreur s'est produite lors de la lecture du fichier {file_path}: {e}")
        return None

def write_text_file(path, text, verif = True):
    if verif :
        if os.path.exists(path):
            pass
        else:  
            with open(path, 'w') as file:
                for string in text:
                    file.write(string + '\n')
    else :
        with open(path,'w') as file :
            for string in text: 
                file.write(string +'\n')

def df2Entity(dataframe: pd.DataFrame, model = None) -> list:
    results = [helpers.prompts.entity_extract_prompt(text, model="llama3:latest") for text in dataframe['text']]
    if results == []:
        print("\n Error : no valid results to concatenate \n")
        return []
    #results = [item for sublist in results for item in sublist]
    print(f'\n \n{results}\n \n')
    entities = []
    for item in results: #nettoyage des résultats
        if isinstance(item, list):
            for subitem in item:
                if 'entity' in subitem:
                    entities.append(subitem['entity'])
                elif 'entities' in subitem:
                    for entity in subitem['entities']:
                        if 'name' in entity:
                            entities.append(entity['name'])
    print(f'RESULTS \n\n{entities}\n\n')
    print(  f' DIM RESULTS \n{len(entities)}\n')
    entities = list(set(entities))
    return entities

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def patch_vs_entity(image):
    overlap = 0.5 
    patch_size_start = 480
    patch_entity = []
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=25,
    length_function=len,
    is_separator_regex=False,
)
    for size in range(1):
        start = time.time()
        patchsize = patch_size_start + 20*size
        patches, coord, width, height = utils.crop_image_to_patches(image, patchsize, overlap)
        text = utils.patch2text(patches)
        
        write_text_file(path='text/'+image+str(patchsize)+'.txt',text = text)
        write_text_file(path='current_text/'+image+'.txt',text = text, verif=False)
        
        text = read_text_file('current_text/'+image+'.txt')
        loader = DirectoryLoader('current_text', show_progress=True)
        documents = loader.load()
        pages = splitter.split_documents([documents[0]])
        print("Number of chunks = ", len(pages))
        df = documents2Dataframe(pages)
        print(df.head())
        results = df2Entity(df, model = 'llama3:latest')
        patch_entity.append((np.size(results), patchsize))
        print(f'PATCHSIZE RESULTS : \n \n{(np.size(results), patchsize)}\n \n')
    return patch_entity



L= [(145, 120), (100, 140), (101, 160), (76, 180), (77, 200), (47, 220), (36, 240), (34, 260), (29, 280), (23, 300), (30, 320), (9, 340), (15, 360), (11, 380), (16, 400), (13, 420), (19, 440), (9, 460),(13, 480),(14, 500), (5, 520), (6, 540), (8, 560), (3, 580), (6, 600), (7, 620), (5, 640), (7, 660), (7, 680), (0, 700), (2, 720), (0, 740), (0, 760), (2, 780), (0, 800), (0, 820), (0, 840), (0, 860), (3, 880), (0, 900), (3, 920), (2, 940), (0, 960), (1, 980)]

plt.figure(figsize=(10,6))
nombre_entites = [y for (y,x) in L]
patch_size = [x for (y,x) in L]

plt.plot(patch_size, nombre_entites, marker='o', linestyle='-', color='b', label='Nombre d\'entités')
plt.legend()
plt.title('Nombre d\'entités en fonction de la taille des patches')
plt.xlabel('Taille des patches')
plt.ylabel('Nombre d\'entités')
plt.grid(True)
plt.show()

# Tracé des données lissées







#patch_entity = patch_vs_entity('image059.png')



#def experiment(image_path = '/home/aurelien/Bureau/scene_matrices/image059.png'):
#    overlap_start = 0.7
#    patch_size_start = 300
#    for overlap in np.arange(overlap_start, 0.9, 0.10):
#        for patchsize in range(patch_size_start, 526, 75):
#            for clusters in range(2, 5):
#                if f'{overlap}_{patchsize}_{clusters}.png' not in os.listdir('exp/'):
#                    start = time.time()
#                    texts, matsim, mat_clusters = scenes_matrices.scene_matrice(clusters, image_path, patch_size = patchsize, overlap_percent = overlap)
#                    scenes_matrices.display(auth_img_path = image_path, nb_clusters=clusters, mask_clusters = np.transpose(mat_clusters), overlap = overlap, patch_size= patchsize)
#                    print(time.time()-start)
                
#experiment()
