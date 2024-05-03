#https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
from scipy.spatial.distance import cosine
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to("cuda")
