import pandas as pd

df = pd.read_csv("C:\\Users\\aakas\\OneDrive\\Desktop\\Coding\\Python\\PyTorch\\news.csv")
df.drop("Unnamed: 0",axis=1,inplace=True)

import torch
from torch.nn.utils.rnn import pad_sequence
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocessing(dataset,col):
    def safe_tokenize(text):
        text = re.sub(r'[^a-zA-Z0-9 ]'," ",text.lower()).lower().split()
        return text
    def safe_numericalize(tokens):
        return torch.tensor([vocab.get(word,vocab['<unk>']) for word in tokens],dtype=torch.long)
    tokenized = dataset[col].apply(safe_tokenize)
    vocab = {"<pad>":0, "<unk>":1}
    for sentence in tokenized:
        for word in sentence:
            if word not in vocab:
                vocab[word]=len(vocab)
    pad = pad_sequence(tokenized.apply(safe_numericalize),batch_first=True,padding_value =vocab['<unk>'])
    return pad, vocab

padded,vocab = preprocessing(df,"text")
padded2, _  = preprocessing(df,"title")
combined_pad = torch.cat((padded,padded2),dim=1)

from torch.utils.data import DataLoader,Dataset,random_split

class Custom_Dataset(Dataset):
    def __init__(self,text,labels):
        self.text = text
        self.labels = labels
    def __len__(self):
        return len(self.text)
    def __getitem__(self,idx):
        return self.text[idx],self.labels[idx]

def label_encode(text):
    return 0 if text == "FAKE" else 1

labels = torch.tensor(df['label'].apply(label_encode),dtype=torch.long)
dataset = Custom_Dataset(combined_pad,labels)
test_size = int(0.2*len(dataset))
train_size=len(dataset)-test_size
train,test = random_split(dataset,[train_size,test_size])
train = DataLoader(train,batch_size=32,shuffle=True)
test = DataLoader(test,batch_size=32,shuffle=True)
min_loss = float('inf')
best_weight = None

import torch.nn as nn
class NeuralNetwork(nn.Module):
    def __init__(self,vocab_size,embed_size,hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embed_size,padding_idx =0)
        self.lstm = nn.LSTM(embed_size,hidden_size,batch_first=True,bidirectional=True)
        self.l1 = nn.Linear(hidden_size*2,1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        embedded = self.embedding(x)
        _,(hidden,_) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2],hidden[-1]),dim=1)
        hidden = self.relu(hidden)
        return self.sigmoid(self.l1(hidden))

def train_model(model,train,test):   
    learning_rate = 0.001
    num_epochs = 20
    model = NeuralNetwork(vocab_size=len(vocab),embed_size=64,hidden_size=100).to(device=device)
    loss_fn = nn.BCELoss()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    min_loss = float('inf')
    best_weight = None
    for epochs in range(num_epochs):
        model.train()
        total_loss = 0
        for text, label in train:
            text, label = text.to(device), label.to(device)
            out = model(text)
            label = label.view(-1,1).float()
            loss = loss_fn(out,label)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
        print(f'Epoch: {epochs+1}/{num_epochs} | Loss: {total_loss/len(train):.3f}')
        with torch.no_grad():
                total_loss = 0.0
                correct = 0
                total = 0
                for data, target in test:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    target = target.view(-1, 1).float()
                    loss = loss_fn(output, target)
                    total_loss += loss.item() * data.size(0)
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
                
                avg_loss = total_loss / len(test)
                accuracy = 100 * correct / total
                
                if avg_loss < min_loss:
                    min_loss = avg_loss
                    best_weight = model.state_dict()
    model.load_state_dict(best_weight)
    return f'Training complete. Best loss: {min_loss:.4f}, Accuracy: {accuracy:.2f}%'
    
model = NeuralNetwork(vocab_size=len(vocab), embed_size=64, hidden_size=100).to(device)
print(train_model(model,train,test))

torch.save(model.state_dict(), "model.pth")