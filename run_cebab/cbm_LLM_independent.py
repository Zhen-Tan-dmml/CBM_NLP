import torch
import transformers
from transformers import RobertaTokenizer, RobertaModel,BertModel, BertTokenizer,GPT2Model, GPT2Tokenizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd 
import os 
from cbm_template_models import MLP, FC
from cbm_models import ModelXtoC_function, ModelCtoY_function
from torch.optim.lr_scheduler import StepLR

# Enable concept or not
mode = 'independent'

# Define the paths to the dataset and pretrained model
# model_name = "microsoft/roberta-base"
model_name = 'bert-base-uncased' # 'bert-base-uncased' / 'roberta-base' / 'gpt2'

# Load the tokenizer and pretrained model
if model_name == 'roberta-base':
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name)
elif model_name == 'bert-base-uncased':
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
elif model_name == 'gpt2':
    model = GPT2Model.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

# Define the maximum sequence length and batch size
max_len = 128
batch_size = 8
is_aux_logits = False
num_labels = 5  #label的个数              
num_each_concept_classes  = 3  #每个concept有几个类
num_epochs = 20

data_type = "pure_cebab" # "pure_cebab"/"aug_cebab"/"aug_yelp"/"aug_cebab_yelp"
# Load data
if data_type == "pure_cebab":
    num_concept_labels = 4
    train_split = "train_exclusive"
    test_split = "test"
    CEBaB = load_dataset("CEBaB/CEBaB")
elif data_type == "aug_cebab":
    num_concept_labels = 10
    train_split = "train_aug_cebab"
    test_split = "test_aug_cebab"
    CEBaB = {}
    CEBaB[train_split] = pd.read_csv("../../dataset/cebab/train_cebab_new_concept_single.csv")
    CEBaB[test_split] = pd.read_csv("../../dataset/cebab/test_cebab_new_concept_single.csv")
elif data_type == "aug_yelp":
    num_concept_labels = 10
    train_split = "train_aug_yelp"
    test_split = "test_aug_yelp"
    CEBaB = {}
    CEBaB[train_split] = pd.read_csv("../../dataset/cebab/train_yelp_new_concept_single.csv")
    CEBaB[test_split] = pd.read_csv("../../dataset/cebab/test_yelp_new_concept_single.csv")
elif data_type == "aug_cebab_yelp":
    num_concept_labels = 10

    train_split = "train_aug_cebab_yelp"
    test_split = "test_aug_cebab_yelp"
    train_split_cebab = pd.read_csv("../../dataset/cebab/train_cebab_new_concept_single.csv")
    test_split_cebab = pd.read_csv("../../dataset/cebab/test_cebab_new_concept_single.csv")
    train_split_yelp = pd.read_csv("../../dataset/cebab/train_yelp_new_concept_single.csv")
    test_split_yelp = pd.read_csv("../../dataset/cebab/test_yelp_new_concept_single.csv")

    CEBaB = {}
    CEBaB[train_split] = pd.concat([train_split_cebab, train_split_yelp], ignore_index=True)
    CEBaB[test_split] = pd.concat([test_split_cebab, test_split_yelp], ignore_index=True)

# Define a custom dataset class for loading the data
class MyDataset(Dataset):
    # Split = train/dev/test
    def __init__(self, split, skip_class = "no majority"):
        self.data = CEBaB[split]
        self.labels = self.data["review_majority"]
        self.text = self.data["description"]
       
        self.food_aspect = self.data["food_aspect_majority"]
        self.ambiance_aspect = self.data["ambiance_aspect_majority"]
        self.service_aspect = self.data["service_aspect_majority"]
        self.noise_aspect =self.data["noise_aspect_majority"]

        if data_type != "pure_cebab":
            # cleanliness price	location	menu variety	waiting time	waiting area	## parking	wi-fi	kids-friendly
            self.cleanliness_aspect = self.data["cleanliness"]
            self.price_aspect = self.data["price"]
            self.location_aspect = self.data["location"]
            self.menu_variety_aspect = self.data["menu variety"]
            self.waiting_time_aspect =self.data["waiting time"]
            self.waiting_area_aspect =self.data["waiting area"]

        self.map_dict = {"Negative":0, "Positive":1, "unknown":2, "":2,"no majority":2}

        self.skip_class = skip_class
        if skip_class is not None:
            self.indices = [i for i, label in enumerate(self.labels) if label != skip_class]
        else:
            self.indices = range(len(self.labels))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        text = self.text[self.indices[index]]
        label = int(self.labels[self.indices[index]]) - 1
        food_concept = self.map_dict[self.food_aspect[self.indices[index]]]
        ambiance_concept = self.map_dict[self.ambiance_aspect[self.indices[index]]]
        service_concept = self.map_dict[self.service_aspect[self.indices[index]]]
        noise_concept = self.map_dict[self.noise_aspect[self.indices[index]]]

        if data_type != "pure_cebab":
            # noisy labels
            #cleanliness price	location	menu variety	waiting time	waiting area	## parking	wi-fi	kids-friendly
            cleanliness_concept = self.map_dict[self.cleanliness_aspect[self.indices[index]]]
            price_concept = self.map_dict[self.price_aspect[self.indices[index]]]
            location_concept = self.map_dict[self.location_aspect[self.indices[index]]]
            menu_variety_concept = self.map_dict[self.menu_variety_aspect[self.indices[index]]]
            waiting_time_concept = self.map_dict[self.waiting_time_aspect[self.indices[index]]]
            waiting_area_concept = self.map_dict[self.waiting_area_aspect[self.indices[index]]]

        if data_type != "pure_cebab":
            concept_labels = [food_concept,ambiance_concept,service_concept,noise_concept,cleanliness_concept,price_concept,location_concept,menu_variety_concept,waiting_time_concept,waiting_area_concept]
        else: 
            concept_labels = [food_concept,ambiance_concept,service_concept,noise_concept]

        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt"
        )
        if data_type != "pure_cebab":
            return {
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
                "label": torch.tensor(label, dtype=torch.long),
                "food_concept": torch.tensor(food_concept, dtype=torch.long),
                "ambiance_concept": torch.tensor(ambiance_concept, dtype=torch.long),
                "service_concept": torch.tensor(service_concept, dtype=torch.long),
                "noise_concept": torch.tensor(noise_concept, dtype=torch.long),
                "cleanliness_concept": torch.tensor(cleanliness_concept, dtype=torch.long),
                "price_concept": torch.tensor(price_concept, dtype=torch.long),
                "location_concept": torch.tensor(location_concept, dtype=torch.long),
                "menu_variety_concept": torch.tensor(menu_variety_concept, dtype=torch.long),
                "waiting_time_concept": torch.tensor(waiting_time_concept, dtype=torch.long),
                "waiting_area_concept": torch.tensor(waiting_area_concept, dtype=torch.long),
                "concept_labels": torch.tensor(concept_labels, dtype=torch.long)
            }
        else:
            return {
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
                "label": torch.tensor(label, dtype=torch.long),
                "food_concept": torch.tensor(food_concept, dtype=torch.long),
                "ambiance_concept": torch.tensor(ambiance_concept, dtype=torch.long),
                "service_concept": torch.tensor(service_concept, dtype=torch.long),
                "noise_concept": torch.tensor(noise_concept, dtype=torch.long),
                "concept_labels": torch.tensor(concept_labels, dtype=torch.long)
            }


# Load the data
train_dataset = MyDataset(train_split)
# val_dataset = MyDataset('validation')
test_dataset = MyDataset(test_split)



# Define the dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

#Set ModelXtoC_layer and ModelCtoY_layer
ModelXtoC_layer = ModelXtoC_function(num_classes = num_each_concept_classes, n_attributes = num_concept_labels, bottleneck = True, expand_dim = 0,aux_logits=is_aux_logits)

# Set up the optimizer and loss function
# optimizer = torch.optim.AdamW(classifier.parameters(), lr=2e-5)
optimizer = torch.optim.Adam(list(model.parameters()) + list(ModelXtoC_layer.parameters()), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# classifier.to(device)
ModelXtoC_layer.to(device)
model.to(device)

#step 1  XtoC

print("train XtoC!")
for epoch in range(num_epochs):
    predicted_concepts_train = []
    predicted_concepts_train_label = []
    ModelXtoC_layer.train()
    model.train()
    
    for batch in tqdm(train_loader, desc="Training", unit="batch"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label = batch["label"].to(device)
        food_concept = batch["food_concept"].to(device)
        ambiance_concept=batch["ambiance_concept"].to(device)
        service_concept=batch["service_concept"].to(device)
        noise_concept=batch["noise_concept"].to(device)
        if data_type != "pure_cebab":
            cleanliness_concept = batch["cleanliness_concept"].to(device)
            price_concept = batch["price_concept"].to(device)
            location_concept = batch["location_concept"].to(device)
            menu_variety_concept = batch["menu_variety_concept"].to(device)
            waiting_time_concept = batch["waiting_time_concept"].to(device)
            waiting_area_concept = batch["waiting_area_concept"].to(device)                
        concept_labels=batch["concept_labels"].to(device)
        concept_labels = torch.t(concept_labels)
        concept_labels = concept_labels.contiguous().view(-1) 

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(1)
        XtoC_output = ModelXtoC_layer(pooled_output)  #4个 8*4
        XtoC_logits = torch.nn.Sigmoid()(torch.cat(XtoC_output, dim=0)) # 32*4 00000000111111112222222233333333
        loss = loss_fn(XtoC_logits, concept_labels)
        loss.backward()
        optimizer.step()

    model.eval()
    ModelXtoC_layer.eval()
    test_accuracy = 0.
    predict_labels = np.array([])
    true_labels = np.array([])
    labelY = []
    predict_concepts = []

    best_acc_score = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)
            food_concept = batch["food_concept"].to(device)
            ambiance_concept=batch["ambiance_concept"].to(device)
            service_concept=batch["service_concept"].to(device)
            noise_concept=batch["noise_concept"].to(device)
            
            if data_type != "pure_cebab":
                cleanliness_concept = batch["cleanliness_concept"].to(device)
                price_concept = batch["price_concept"].to(device)
                location_concept = batch["location_concept"].to(device)
                menu_variety_concept = batch["menu_variety_concept"].to(device)
                waiting_time_concept = batch["waiting_time_concept"].to(device)
                waiting_area_concept = batch["waiting_area_concept"].to(device)                    
            concept_labels=batch["concept_labels"].to(device)  #8*4
            concept_labels = torch.t(concept_labels) #4*8
            concept_labels = concept_labels.contiguous().view(-1) #4*8=32
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.last_hidden_state.mean(1)
            logits = ModelXtoC_layer(pooled_output)     
            logits = torch.cat(logits, dim=0)
            predictions = torch.argmax(logits, axis=1)
            test_accuracy += torch.sum(predictions == concept_labels).item()
            predict_labels = np.append(predict_labels, predictions.cpu().numpy())
            true_labels = np.append(true_labels, concept_labels.cpu().numpy())
            predictions = predictions.reshape(-1,num_concept_labels)  # reshape 二维向量[batch_size*num_concept_labels]
            predict_concepts.append(predictions)
            labelY.append(label)
        test_accuracy /= len(test_dataset)
        num_true_labels = len(np.unique(true_labels))
        
        macro_f1_scores = []
        for label in range(num_true_labels):
            label_pred = np.array(predict_labels) == label
            label_true = np.array(true_labels) == label
            macro_f1_scores.append(f1_score(label_true, label_pred, average='macro'))
            mean_macro_f1_score = np.mean(macro_f1_scores)

    print(f"Epoch {epoch + 1}: Test Concept Acc = {test_accuracy*100/num_concept_labels} Test Concept Macro F1 = {mean_macro_f1_score*100}")
    if test_accuracy > best_acc_score:
        best_acc_score = test_accuracy
        best_predicted_concepts = predict_concepts
        best_labels = labelY
        torch.save(model, "./"+model_name+"_independent.pth")
        torch.save(ModelXtoC_layer, "./"+model_name+"_ModelXtoC_layer_independent.pth")
                   
#step 2  CtoY
num_epochs = 50
print("train CtoY first, then treat predicted C of XtoC as input at test time!")
#ModelCtoY_layer = ModelCtoY_function(n_class_attr = 0, n_attributes = num_each_concept_classes*num_concept_labels, num_classes = num_labels, expand_dim = 0)
ModelCtoY_layer = ModelCtoY_function(n_attributes = num_each_concept_classes*num_concept_labels, num_classes = num_labels, expand_dim = 0)
model = torch.load("./"+model_name+"_independent.pth")
model = torch.load("./"+model_name+"_independent.pth")
ModelXtoC_layer = torch.load("./"+model_name+"_ModelXtoC_layer_independent.pth") 

# Set up the optimizer and loss function
optimizer = torch.optim.Adam(ModelCtoY_layer.parameters(), lr=1e-3, weight_decay=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()
scheduler = StepLR(optimizer, step_size=15*len(train_loader), gamma=0.5)

# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# classifier.to(device)
ModelCtoY_layer.to(device)

for epoch in range(num_epochs):
    ModelCtoY_layer.train()
    
    for batch in tqdm(train_loader, desc="Training", unit="batch"):
        input_ids = batch["input_ids"].to(device)
        label = batch["label"].to(device)
        food_concept = batch["food_concept"].to(device)
        ambiance_concept=batch["ambiance_concept"].to(device)
        service_concept=batch["service_concept"].to(device)
        noise_concept=batch["noise_concept"].to(device)
        if data_type != "pure_cebab":
            cleanliness_concept = batch["cleanliness_concept"].to(device)
            price_concept = batch["price_concept"].to(device)
            location_concept = batch["location_concept"].to(device)
            menu_variety_concept = batch["menu_variety_concept"].to(device)
            waiting_time_concept = batch["waiting_time_concept"].to(device)
            waiting_area_concept = batch["waiting_area_concept"].to(device)                
        concept_labels=batch["concept_labels"].to(device)
        concept_labels = F.one_hot(concept_labels)
        concept_labels = concept_labels.reshape(-1,num_each_concept_classes*num_concept_labels)
        concept_labels = concept_labels.to(torch.float32)
        optimizer.zero_grad()
        CtoY_logits = ModelCtoY_layer(concept_labels)  #[batch_size,concept_size]     
        CtoY_logits = torch.nn.Sigmoid()(CtoY_logits)        
        loss = loss_fn(CtoY_logits, label)
        loss.backward()
        optimizer.step()
        # adjust learning rate using scheduler
        scheduler.step()
    
    ModelCtoY_layer.eval()
    test_accuracy = 0.
    predict_labels = np.array([])
    true_labels = np.array([])

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)
            food_concept = batch["food_concept"].to(device)
            ambiance_concept=batch["ambiance_concept"].to(device)
            service_concept=batch["service_concept"].to(device)
            noise_concept=batch["noise_concept"].to(device)
            
            if data_type != "pure_cebab":
                cleanliness_concept = batch["cleanliness_concept"].to(device)
                price_concept = batch["price_concept"].to(device)
                location_concept = batch["location_concept"].to(device)
                menu_variety_concept = batch["menu_variety_concept"].to(device)
                waiting_time_concept = batch["waiting_time_concept"].to(device)
                waiting_area_concept = batch["waiting_area_concept"].to(device)                    
            concept_labels=batch["concept_labels"].to(device)  #8*4

            # 用训练好的 x->c model 得预测 concept labels
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.last_hidden_state.mean(1)
            logits = ModelXtoC_layer(pooled_output)     #4个8*3

            logits = torch.stack(logits, dim=0)  #[4,8,3]
            logits=torch.transpose(logits, 0, 1) #[8,4,3]

            # predictions_concept_labels = logits.reshape(-1,num_each_concept_classes*num_concept_labels)  #logits: this line / one-hot:the following four lines 
            predictions_concept_labels = torch.argmax(logits, axis=-1) #[8,4]
            predictions_concept_labels = predictions_concept_labels.reshape(-1,num_concept_labels)  # reshape 二维向量[batch_size*num_concept_labels]
            predictions_concept_labels = F.one_hot(predictions_concept_labels)
            predictions_concept_labels = predictions_concept_labels.reshape(-1,num_each_concept_classes*num_concept_labels)

            predictions_concept_labels = predictions_concept_labels.to(torch.float32)
            CtoY_logits = ModelCtoY_layer(predictions_concept_labels)
            predictions_labels = torch.argmax(CtoY_logits, axis=1)

            test_accuracy += torch.sum(predictions_labels == label).item()
            predict_labels = np.append(predict_labels, predictions_labels.cpu().numpy())
            true_labels = np.append(true_labels, label.cpu().numpy())

        test_accuracy /= len(test_dataset)
        num_true_labels = len(np.unique(true_labels))
        
        macro_f1_scores = []
        for label in range(num_true_labels):
            label_pred = np.array(predict_labels) == label
            label_true = np.array(true_labels) == label
            macro_f1_scores.append(f1_score(label_true, label_pred, average='macro'))
            mean_macro_f1_score = np.mean(macro_f1_scores)


    print(f"Epoch {epoch + 1}: Test Acc = {test_accuracy*100} Test Macro F1 = {mean_macro_f1_score*100}")