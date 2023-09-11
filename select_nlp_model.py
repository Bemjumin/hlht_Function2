import torch
from transformers import AutoTokenizer, BertForSequenceClassification
import datasets

my_dataset_all = datasets.load_dataset(path='sst2', cache_dir='./data')