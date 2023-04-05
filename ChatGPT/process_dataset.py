import torch
import transformers
from transformers import GPT2Model, GPT2Tokenizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import openai
import json
import csv
def load_openai_request():
    #input your key
    openai_key='√√√√√√√√'
    openai.api_key=openai_key


    concepts=['food', 'ambiance', 'service', 'noise']
    # Load data
    #yelp=load_dataset("yelp_review_full")
    #train_text=yelp['train']['text']
    #train_label=yelp['train']['label']
    #test_text=yelp['test']['text']
    #test_label=yelp['test']['label']
    data_file='./Yelp Restaurant Reviews.csv'
    dataset = load_dataset('csv', data_files=data_file)
    selected=np.random.choice(list(dataset['train']), size=3000, replace=False)
    train_text=[one['Review Text'] for one in selected[:2000]]
    train_label=[one['Rating'] for one in selected[:2000]]
    test_text=[one['Review Text'] for one in selected[2000:]]
    test_label=[one['Rating'] for one in selected[2000:]]
    train_data=[]
    for text, label in zip(train_text, train_label):
        data=[text, str(label)]
        for concept in concepts:
            question="according to the review \"{}\", how is the {} of the restaurant? " \
                     "please answer with one option in good, bad, or unknown".format(text, concept)
            response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                    messages=[{"role": "user",
                                                               "content": question}])
            answer=response['choices'][0]['message']['content'].strip('.').lower().strip('\n').strip('\n')
            data.append(answer)
        print(data)
        train_data.append(data)
        if len(train_data)==2000: break
    test_data=[]
    for text, label in zip(test_text, test_label):
        data = [text, str(label)]
        for concept in concepts:
            question = "according to the review \"{}\", how is the {} of the restaurant? " \
                       "please answer with one option in good, bad, or unknown".format(text, concept)
            response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                    messages=[{"role": "user",
                                                               "content": question}])
            answer = response['choices'][0]['message']['content'].strip('.').lower().strip('\n').strip('\n')
            data.append(answer)
        test_data.append(data)
        if len(test_data)==1000: break
    train_data=pd.DataFrame(train_data, columns=['description', 'review_majority', 'food_aspect_majority',
                                                 'ambiance_aspect_majority', 'service_aspect_majority',
                                                 'noise_aspect_majority'])
    train_data.to_csv('./train.csv')
    test_data=pd.DataFrame(test_data, columns=['description', 'review_majority', 'food_aspect_majority',
                                                 'ambiance_aspect_majority', 'service_aspect_majority',
                                                 'noise_aspect_majority'])
    test_data.to_csv('./test.csv')
    dataset = load_dataset('csv', data_files={'train': './train.csv',
                                                  'test': './test.csv'})
    dataset = dataset.rename_column(
        original_column_name="Unnamed: 0", new_column_name="id"
    )

    print(dataset)
    
load_openai_request()
