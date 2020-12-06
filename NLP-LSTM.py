import pandas as pd
from string import punctuation
import numpy as np
import torch
from nltk.tokenize import word_tokenize
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from torch import optim
import json

class SentimentLSTM(nn.Module):
    
    def __init__(self, n_vocab, n_embed, n_hidden, n_output, n_layers, drop_p = 0.8):
        super().__init__()
        
        self.n_vocab = n_vocab  
        self.n_layers = n_layers 
        self.n_hidden = n_hidden 
        
        self.embedding = nn.Embedding(n_vocab, n_embed)
        self.lstm = nn.LSTM(n_embed, n_hidden, n_layers, batch_first = True, dropout = drop_p)
        self.dropout = nn.Dropout(drop_p)
        self.fc = nn.Linear(n_hidden, n_output)
        self.sigmoid = nn.Sigmoid()
        
        
    def forward (self, input_words, hidden):
        embedded_words = self.embedding(input_words.cuda())
        lstm_out, h = self.lstm(embedded_words, hidden) 
        lstm_out = self.dropout(lstm_out)
        lstm_out = lstm_out.contiguous().view(-1, self.n_hidden)
        fc_out = self.fc(lstm_out)                  
        sigmoid_out = self.sigmoid(fc_out)              
        sigmoid_out = sigmoid_out.view(batch_size, -1)  
        
        sigmoid_last = sigmoid_out[:, -1]
        
        return sigmoid_last, h
    
    
    def init_hidden (self, batch_size):
        
        device = "cuda"
        weights = next(self.parameters()).data
        h = (weights.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device),
            weights.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device))
        return h


def split_words_reviews(data):
    text = list(data['Twitter'].values)
    clean_text = []
    for t in text:
        clean_text.append(t.translate(str.maketrans('', '', punctuation)).lower().rstrip())
    tokenized = [word_tokenize(x) for x in clean_text]
    all_text = []
    for tokens in tokenized:
        for t in tokens:
            all_text.append(t)
    return tokenized, set(all_text)

def create_dictionaries(words):
    word_to_int_dict = {w:i+1 for i, w in enumerate(words)}
    int_to_word_dict = {i:w for w, i in word_to_int_dict.items()}
    return word_to_int_dict, int_to_word_dict

def pad_text(tokenized_twitts, seq_length):

    twitts = []

    for tweet in tokenized_twitts:

        if len(tweet) >= seq_length:
            twitts.append(tweet[:seq_length])
        else:

            twitts.append(['']*(seq_length-len(tweet)) + tweet)

    print(twitts[6478])
    return np.array(twitts,dtype=object)

with open("proccessedText2.txt") as f:
    twitts = f.read()

data = pd.DataFrame([twitts.split('\t') for twitts in twitts.split('\n')])

data.columns = ['Twitter','label','Location','Type']

data = data.sample(frac=1)

print(data)


twitts, vocab = split_words_reviews(data)

word_to_int_dict, int_to_word_dict = create_dictionaries(vocab)

twitts_padded = pad_text(twitts, seq_length = np.max([len(x) for x in twitts]))

int_to_word_dict[0] = ''
word_to_int_dict[''] = 0

test = [[word_to_int_dict[word] for word in twitt] for twitt in twitts_padded]

encoded_sentences = np.array([[word_to_int_dict[word] for word in twitt] for twitt in twitts_padded])


labels = np.array([int(x) for x in data['label'].values])



print(twitts[0])
print(encoded_sentences[0])
print(labels[0])

n_vocab = len(word_to_int_dict)
n_embed = 50
n_hidden = 100
n_output = 1
n_layers = 2
train_ratio = 0.8
valid_ratio = (1 - train_ratio)/2
batch_size = 1


net = SentimentLSTM(n_vocab, n_embed, n_hidden, n_output, n_layers).cuda()

total = len(encoded_sentences)
train_cutoff = int(total * train_ratio)
valid_cutoff = int(total * (1 - valid_ratio))

train_x, train_y = torch.Tensor(encoded_sentences[:train_cutoff]).long().cuda(), torch.Tensor(labels[:train_cutoff]).long().cuda()
valid_x, valid_y = torch.Tensor(encoded_sentences[train_cutoff : valid_cutoff]).long().cuda(), torch.Tensor(labels[train_cutoff : valid_cutoff]).long().cuda()
test_x, test_y = torch.Tensor(encoded_sentences[valid_cutoff:]).long().cuda(), torch.Tensor(labels[valid_cutoff:]).long().cuda()

train_data = TensorDataset(train_x, train_y)
valid_data = TensorDataset(valid_x, valid_y)
test_data = TensorDataset(test_x, test_y)

train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
valid_loader = DataLoader(valid_data, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = True)

print_every = 800
step = 0
n_epochs = 3
clip = 5
criterion = nn.BCELoss().cuda()
optimizer = optim.Adam(net.parameters(), lr = 0.001)

count=0

for epoch in range(n_epochs):
    
    for inputs, labels in train_loader:
        step += 1  
        net.zero_grad()
        #Las capas ocultas las maneja de forma interna y no se dan como input. Si esta vacia se inicializa sola
        output, h = net(inputs)
        loss = criterion(output.view(-1).cuda(), labels.float().view(-1).cuda())
        if(count==0):
            loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()
        if (step % print_every) == 0:
            print(h)

            net.eval()
            valid_losses = []
            
            for v_inputs, v_labels in valid_loader:
                       
                v_output, v_h = net(v_inputs)
                v_loss = criterion(v_output.squeeze().view(-1).cuda(), v_labels.float().view(-1).cuda())
                valid_losses.append(v_loss.item())

            print("Epoch: {}/{}".format((epoch+1), n_epochs),
                  "Step: {}".format(step),
                  "Training Loss: {:.4f}".format(loss.item()),
                  "Validation Loss: {:.4f}".format(np.mean(valid_losses)))
            net.train()


net.eval()
test_losses = []
num_correct = 0



for inputs, labels in test_loader:

    test_output, test_h = net(inputs)
    loss = criterion(test_output.float().view(-1).cuda(), labels.float().view(-1).cuda())
    test_losses.append(loss.item())
    
    preds = torch.round(test_output.squeeze())
    correct_tensor = preds.eq(labels.float().view_as(preds))
    
    correct = np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)
    
print("Test Loss: {:.4f}".format(np.mean(test_losses)))
print("Test Accuracy: {:.2f}".format(num_correct/len(test_loader.dataset)))

def preprocess_tweet(tweet):
    tweet = tweet.translate(str.maketrans('', '', punctuation)).lower().rstrip()
    tokenized = word_tokenize(tweet)
    if len(tokenized) >= 50:
        tweet = tokenized[:50]
    else:
        tweet= ['0']*(50-len(tokenized)) + tokenized
    
    final = []
    
    for token in tweet:
        try:
            final.append(word_to_int_dict[token])
            
        except:
            final.append(word_to_int_dict[''])
        
    return final

def predict(tweet):
    net.eval()
    words = np.array([preprocess_tweet(tweet)])
    padded_words = torch.from_numpy(words).cuda()
    pred_loader = DataLoader(padded_words, batch_size = 1, shuffle = True)
    for x in pred_loader:
        print(x.size())
        print(x.type())
        output = net(x.type(torch.long))[0].item()
    
    msg = "Este tweet es sobre un desastre" if output >= 0.5 else "Este tweet no es sobre un desastre"
    print(msg)
    print(tweet)
    print('Prediccion = ' + str(output))


print(predict("Thoughts and prayers going out to the people of Nepal F"))

while True:
    print('\n')
    print(predict(input("Ingrese Prediccion: ")))
    print('\n')