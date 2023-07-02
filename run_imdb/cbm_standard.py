import torch
import transformers
from gensim.models import FastText
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from transformers import RobertaTokenizer, RobertaModel,BertModel, BertTokenizer,GPT2Model, GPT2Tokenizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd

# Enable concept or not
mode = 'standard'

# Define the paths to the dataset and pretrained model
# model_name = "microsoft/deberta-base"
model_name = 'bert-base-uncased' # 'bert-base-uncased' / 'roberta-base' / 'gpt2' / 'lstm'

# Define the maximum sequence length, batch size, num_concepts_size,num_labels,num_epochs
max_len = 128
batch_size = 8
num_labels = 2 
num_epochs = 1

# Load the tokenizer and pretrained model
if model_name == 'roberta-base':
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name)
elif model_name == 'bert-base-uncased':
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
elif model_name == 'gpt2':
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  
    model = GPT2Model.from_pretrained(model_name)
elif model_name == 'lstm':
    fasttext_model = FastText.load_fasttext_format('../fasttext/cc.en.300.bin')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    class BiLSTMWithDotAttention(torch.nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
            embeddings = fasttext_model.wv.vectors
            self.embedding.weight = torch.nn.Parameter(torch.tensor(embeddings))
            self.embedding.weight.requires_grad = False
            self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers = 1, bidirectional=True, batch_first=True)

        def forward(self, input_ids, attention_mask):
            input_lengths = attention_mask.sum(dim=1)
            embedded = self.embedding(input_ids)
            output, _ = self.lstm(embedded)
            weights = F.softmax(torch.bmm(output, output.transpose(1, 2)), dim=2)
            attention = torch.bmm(weights, output)
            return attention

    model = BiLSTMWithDotAttention(len(tokenizer.vocab), 300, 128, num_labels)
   

data_type = "manual_imdb" # "manual_imdb"/"aug_manual_imdb"/"gen_imdb"/"aug_gen_imdb"
# Load data
if data_type == "manual_imdb":
    num_concept_labels = 4
    train_split = "manual_imdb"
    test_split = "manual_imdb_test"
    val_split = "manual_imdb_val"
    CEBaB = {}
    CEBaB[train_split] = pd.read_csv("../dataset/imdb/IMDB-train-manual.csv")
    CEBaB[test_split] = pd.read_csv("../dataset/imdb/IMDB-test-manual.csv")
    CEBaB[val_split] = pd.read_csv("../dataset/imdb/IMDB-dev-manual.csv")
elif data_type == "aug_manual_imdb":
    num_concept_labels = 8
    train_split = "aug_manual_imdb"
    test_split = "aug_manual_imdb_test"
    val_split = "manual_imdb_val"
    CEBaB = {}
    CEBaB[train_split] = pd.read_csv("../dataset/imdb/IMDB-train-manual.csv")
    CEBaB[test_split] = pd.read_csv("../dataset/imdb/IMDB-test-manual.csv")
    CEBaB[val_split] = pd.read_csv("../dataset/imdb/IMDB-dev-manual.csv")
elif data_type == "gen_imdb":
    num_concept_labels = 8
    train_split = "gen_imdb"
    test_split = "gen_imdb_test"
    val_split = "gen_imdb_val"
    CEBaB = {}
    CEBaB[train_split] = pd.read_csv("../dataset/imdb/IMDB-train-generated.csv")
    CEBaB[test_split] = pd.read_csv("../dataset/imdb/IMDB-test-generated.csv")
    CEBaB[val_split] = pd.read_csv("../dataset/imdb/IMDB-dev-generated.csv")
elif data_type == "aug_gen_imdb":
    num_concept_labels = 8
    train_split = "aug_gen_imdb"
    test_split = "aug_gen_imdb_test"
    val_split = "aug_gen_imdb_val"
    train_split_manual= pd.read_csv("../dataset/imdb/IMDB-train-manual.csv")
    test_split_manual = pd.read_csv("../dataset/imdb/IMDB-test-manual.csv")
    val_split_manual = pd.read_csv("../dataset/imdb/IMDB-dev-manual.csv")
    train_split_generated = pd.read_csv("../dataset/imdb/IMDB-train-generated.csv")
    test_split_generated = pd.read_csv("../dataset/imdb/IMDB-test-generated.csv")
    val_split_generated = pd.read_csv("../dataset/imdb/IMDB-dev-generated.csv")

    CEBaB = {}
    CEBaB[train_split] = pd.concat([train_split_manual, train_split_generated], ignore_index=True)
    CEBaB[test_split] = pd.concat([test_split_manual, test_split_generated], ignore_index=True)
    CEBaB[val_split] = pd.concat([val_split_manual, val_split_generated], ignore_index=True)
# Define a custom dataset class for loading the data

class MyDataset(Dataset):
    # Split = train/dev/test
    def __init__(self, split, skip_class = "no majority"):
        self.data = CEBaB[split]
        self.labels = self.data["sentiment"]
        self.text = self.data["review"]
       
        self.acting_aspect = self.data["acting"]
        self.storyline_aspect = self.data["storyline"]
        self.emotional_aspect = self.data["emotional arousal"]
        self.cinematography_aspect =self.data["cinematography"]

        if data_type != "manual_imdb":
            # soundtrack	directing	background setting	editing
            self.soundtrack_aspect = self.data["soundtrack"]
            self.directing_aspect = self.data["directing"]
            self.background_aspect = self.data["background setting"]
            self.editing_aspect = self.data["editing"]


        self.map_dict = {"Negative":0, "Negative ":0, "Positive":1, "unknown":2,"Unkown":2}

        self.skip_class = skip_class
        if skip_class is not None:
            self.indices = [i for i, label in enumerate(self.labels) if label != skip_class]
        else:
            self.indices = range(len(self.labels))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        text = self.text[self.indices[index]]
        label_dict = {"Positive": 1, "Negative": 0}
        label = label_dict[self.labels[self.indices[index]]]

        # gold labels
        #acting	 storyline	emotional arousal	cinematography	

        acting_concept = self.map_dict[self.acting_aspect[self.indices[index]].strip()]
        storyline_concept = self.map_dict[self.storyline_aspect[self.indices[index]].strip()]
        emotional_concept = self.map_dict[self.emotional_aspect[self.indices[index]].strip()]
        cinematography_concept = self.map_dict[self.cinematography_aspect[self.indices[index]].strip()]
        
        if data_type != "manual_imdb":
            # noisy labels
            #soundtrack	directing	background setting	editing
            soundtrack_concept = self.map_dict[self.soundtrack_aspect[self.indices[index]].strip()]
            directing_concept = self.map_dict[self.directing_aspect[self.indices[index]].strip()]
            background_concept = self.map_dict[self.background_aspect[self.indices[index]].strip()]
            editing_concept = self.map_dict[self.editing_aspect[self.indices[index]].strip()]


        if data_type != "manual_imdb":
            concept_labels = [acting_concept,storyline_concept,emotional_concept,cinematography_concept,soundtrack_concept,directing_concept,background_concept,editing_concept]
        else: 
            concept_labels = [acting_concept,storyline_concept,emotional_concept,cinematography_concept]

        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt"
        )
        if data_type != "manual_imdb":
            return {
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
                "label": torch.tensor(label, dtype=torch.long),

                "acting_concept": torch.tensor(acting_concept, dtype=torch.long),
                "storyline_concept": torch.tensor(storyline_concept, dtype=torch.long),
                "emotional_concept": torch.tensor(emotional_concept, dtype=torch.long),
                "cinematography_concept": torch.tensor(cinematography_concept, dtype=torch.long),

                "soundtrack_concept": torch.tensor(soundtrack_concept, dtype=torch.long),
                "directing_concept": torch.tensor(directing_concept, dtype=torch.long),
                "background_concept": torch.tensor(background_concept, dtype=torch.long),
                "editing_concept": torch.tensor(editing_concept, dtype=torch.long),

                "concept_labels": torch.tensor(concept_labels, dtype=torch.long)
            }
        else:
            return {
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
                "label": torch.tensor(label, dtype=torch.long),
                "acting_concept": torch.tensor(acting_concept, dtype=torch.long),
                "storyline_concept": torch.tensor(storyline_concept, dtype=torch.long),
                "emotional_concept": torch.tensor(emotional_concept, dtype=torch.long),
                "cinematography_concept": torch.tensor(cinematography_concept, dtype=torch.long),
                "concept_labels": torch.tensor(concept_labels, dtype=torch.long)
            }


# Load the data
train_dataset = MyDataset(train_split)
test_dataset = MyDataset(test_split)
val_dataset = MyDataset(val_split)

# Define the dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# normal
if model_name == 'lstm':
    classifier = torch.nn.Sequential(
        torch.nn.Linear(model.hidden_dim*2, model.hidden_dim),
        torch.nn.Linear(model.hidden_dim, num_concept_labels),
        torch.nn.Linear(num_concept_labels, num_labels)
    )
else:
    classifier = torch.nn.Sequential(
        torch.nn.Linear(model.config.hidden_size, num_concept_labels),
        torch.nn.Linear(num_concept_labels, num_labels)
)


# Set up the optimizer and loss function
optimizer = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=1e-5)
if model_name == 'lstm':
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
loss_fn = torch.nn.CrossEntropyLoss()

# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier.to(device)
model.to(device)

for epoch in range(num_epochs):
    classifier.train()
    model.train()
    
    for batch in tqdm(train_loader, desc="Training", unit="batch"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        if model_name == 'lstm':
            pooled_output = outputs.mean(1) 
        else:
            pooled_output = outputs.last_hidden_state.mean(1)        
        logits = classifier(pooled_output)
        loss = loss_fn(logits, label)
        loss.backward()
        optimizer.step()
    
    model.eval()
    classifier.eval()
    test_accuracy = 0.
    val_accuracy = 0.
    best_acc_score = 0
    predict_labels = np.array([])
    true_labels = np.array([])
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Val", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            if model_name == 'lstm':
                pooled_output = outputs.mean(1) 
            else:
                pooled_output = outputs.last_hidden_state.mean(1)   
            logits = classifier(pooled_output)
            predictions = torch.argmax(logits, axis=1)
            val_accuracy += torch.sum(predictions == label).item()
            predict_labels = np.append(predict_labels, predictions.cpu().numpy())
            true_labels = np.append(true_labels, label.cpu().numpy())
        
        val_accuracy /= len(val_dataset)
        num_true_labels = len(np.unique(true_labels))
        macro_f1_scores = []
        for label in range(num_true_labels):
            label_pred = np.array(predict_labels) == label
            label_true = np.array(true_labels) == label
            macro_f1_scores.append(f1_score(label_true, label_pred, average='macro'))
            mean_macro_f1_score = np.mean(macro_f1_scores)

    print(f"Epoch {epoch + 1}: Val Acc = {val_accuracy*100} Val Macro F1 = {mean_macro_f1_score*100}")
    if val_accuracy > best_acc_score:
        best_acc_score = val_accuracy
        torch.save(classifier, "./"+model_name+"_classifier_standard.pth")
        torch.save(model, "./"+model_name+"_model_standard.pth")

####################### test
num_epochs = 1
print("Test!")
model = torch.load("./"+model_name+"_model_standard.pth")
classifier = torch.load("./"+model_name+"_classifier_standard.pth") 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for epoch in range(num_epochs):
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            if model_name == 'lstm':
                pooled_output = outputs.mean(1) 
            else:
                pooled_output = outputs.last_hidden_state.mean(1)   
            logits = classifier(pooled_output)
            predictions = torch.argmax(logits, axis=1)
            test_accuracy += torch.sum(predictions == label).item()
            predict_labels = np.append(predict_labels, predictions.cpu().numpy())
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