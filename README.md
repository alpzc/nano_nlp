# NanoGPT Training

The execution of this code was done inside a containerized environment using the following docker image: 
```
https://hub.docker.com/r/nablascom/cuda-pytorch
```

The training of the transformer model is done using a standard Jupyter notebook connected through the container in WSL2.

## Python and Data Setup

The following downloads are needed for the training of the model.

### Training Text Download
Text used for training will use the english translation for Don Quixote by Miguel de Cervantes Saavedra.
```
 sudo wget -O don_quixote.txt https://raw.githubusercontent.com/GITenberg/Don-Quixote_996/master/996.txt
```

For more complex data, a sample of Wikipedia articles can be downloaded and uncompressed using:
```
 sudo wget -O wiki_eng.txt.bz2 https://dumps.wikimedia.org/enwiki/20230820/enwiki-20230820-pages-articles-multistream-index.txt.bz2
 sudo bzip2 -d wiki_eng.txt.bz2
```

### Embedding Models
For the word embeddings that are going to be used in the text files, two options can be downloaded: 

* Wiki2Vec
```
sudo wget -O wiki2vec.txt.bz2 http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_300d.txt.bz2
```

* Google Word2Vec
```
sudo wget -O googleNews-vectors-negative300.bin.gz https://raw.githubusercontent.com/mmihaltz/word2vec-GoogleNews-vectors/master/GoogleNews-vectors-negative300.bin.gz
```

* Compilation of other embedding models [here](https://github.com/3Top/word2vec-api#where-to-get-a-pretrained-models)

### Packages Libraries
Inside the notebook, run these before executing any of the other cells in the code:

```
!python -m pip install traitlets==4.3.3 --force-reinstall
!pip install pywin32==228
!pip install tiktoken
!pip install gensim
!pip install numpy==1.21.6
```


## Notebook Code
Run the following code to start processing the data and training the model.
### Importing Packages

```
import torch
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re
import time
```

### Data Preprocessing
* Opening the .txt file and cleaning up using regular expressions
```
with open('./data/don_quixote.txt', "r", encoding = 'utf-8') as f:
    text = f.read()
    
text= text.replace('\n', ' ').replace('  ', ' ')[121506:]

text = text.lower()
```

* Defining function that separats words and special characters and returns them as a list
```
def split_into_words_special_chars_and_spaces(text):
    return re.findall('\w+|\S|\s', text)

text_separated = split_into_words_special_chars_and_spaces(text)

print(text_separated[:100])
```

* Defining the vocabulary size 
```
vocabulary = list(set(text_separated))
chars = sorted(list(set(text)))

char_vocab_size = len(chars)
words_vocab_size = len(vocabulary)

print('Unique characters: ', char_vocab_size)
print('Unique words and special characters: ', words_vocab_size)
```

### Data Tokenization
* Standard unique tokenization for eacch unique word and special character:
```
stoi = { ch:i for i,ch in enumerate(vocabulary) }
itos = { i:ch for i,ch in enumerate(vocabulary) }

def encode(text_string):
    sep_text =split_into_words_special_chars_and_spaces(text_string)
    return [stoi[i] for i in sep_text]

def decode(encoded_text_list):
    decoded_text = [itos[i] for i in encoded_text_list]
    return ''.join(decoded_text)
    

print(encode(text[100:200]))
print(decode(encode(text[100:200])))
```

### Word Embedding
To have each word and special character represented as a numerical vector while retaining semantic meaning using context.

We will be using the Word2Vec model and train it on our text.
To train the embedding model, we first separate the text into a list of lists where the first dimension represents entire sentences and the second dimension represents the words/special characters within each sentence.

```
sentences = []
start = 0
for i, word in enumerate(text_separated):
    if word in [".", '?', '!']:
        sentences.append(text_separated[start : i+1])
        start = i+1
```

The training of the model (documentation [here](https://radimrehurek.com/gensim/models/word2vec.html)) is done using:

```
emb_model = Word2Vec(sentences=sentences,
                 min_count=1,
                 sg=1, 
                 vector_size =500,  
                 workers=4)
```

Testing the embeddings and visulizing it using TSNE plot: 

```
print(emb_model.wv.most_similar('food'))


words = list(emb_model.wv.key_to_index)
vectors = emb_model.wv.vectors

tsne = TSNE(n_components=2, random_state=0)
vectors_2d = tsne.fit_transform(vectors[:2500])

plt.figure(figsize=(10,10))
plt.scatter(vectors_2d[:,0], vectors_2d[:,1], edgecolors='k', c='r')
plt.show()
```

### Defining the Model and Functions

The model follows the architecture in the following form:

<div>
  <pre>
Input Tokens
    |
Token Embedding
    |
    V
Position Embedding
    |
    V
Transformer Block 1 <span style="font-size:20px;">→</span> Transformer Block 2 <span style="font-size:20px;">→</span> ... <span style="font-size:20px;">→</span> Transformer Block N
    |                       |                                   |
    V                       V                                   V
Layer Normalization 1     Layer Normalization 2               Layer Normalization N
    |                       |                                   |
    V                       V                                   V
Residual Connection 1     Residual Connection 2               Residual Connection N
    |                       |                                   |
    V                       V                                   V
Final Layer Normalization
    |
    V
Linear Layer (lm_head)
    |
    V
Batch Normalization
    |
    V
Output (Logits)
  </pre>
</div>


Where the transformer block consists only of the decoder part of the proposed transformer architecture in its original paper [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf). 

<div style="text-align:center;">
  <pre>
Input
    |
    V
Self-Attention
    |
    V
Residual Connection 1
    |
    V
Layer Normalization 1
    |
    V
Feed-Forward Network
    |
    V
Residual Connection 2
    |
    V
Layer Normalization 2
    |
    V
Output
  </pre>
</div>


* Defining the Multihead Attention and Transformer Blocks:
```
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_random_batch(split, block_size = block_size, batch_size= batch_size)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear( n_embed, head_size, bias=False)
        self.query = nn.Linear( n_embed, head_size, bias=False)
        self.value = nn.Linear( n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed,  n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        proj = self.proj(out)
        out = self.dropout(proj)
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self,  n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear( n_embed, 4 *  n_embed),
            nn.ReLU(),
            nn.Linear(4 *  n_embed,  n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self,  n_embed, n_head):
        #  n_embed: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size =  n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward( n_embed)
        self.ln1 = nn.LayerNorm( n_embed)
        self.ln2 = nn.LayerNorm( n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
```

* Defining the Model Overview:
```
class BigramModel(nn.Module):

    def __init__(self, char_vocab_size, n_embed):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(char_vocab_size,  n_embed)
        self.position_embedding_table = nn.Embedding(block_size,  n_embed)
        self.blocks = nn.Sequential(*[Transformer( n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm( n_embed) # final layer norm
        self.lm_head = nn.Linear( n_embed, char_vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,char_vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
```


### Training the Model
The hyperparameters can be changed to train the network.
Time of training will vary heavily on amount of CUDA cores available and amount of parameters in the model.

```
epoch_list = []
train_loss = []
val_loss = []


# hyperparameters
data  = torch.tensor(encode(text), dtype=torch.long)
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 128 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 20
n_embed = 512
n_head = 32
n_layer = 32
dropout = 0.2
vocab_size = len(vocabulary)
# n_embed // n_head must equal batch_size

#defining model
import os.path
if os.path.exists('shared/project_models/gpt_from_scratch/model_backup.pth'):
    model = torch.load('shared/project_models/gpt_from_scratch/model_backup.pth')
else:
    model = BigramModel(vocab_size, n_embed)
m = model.to(device)
print('Model Defined')

# Setting the training split to 90%
train_split = int(0.9 * len(data))
train_data = data[:train_split]
val_data = data[train_split:]

print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')


optimizer = torch.optim.AdamW(m.parameters(), lr =learning_rate)

for step in range(max_iters):
    
    # every once in a while evaluate the loss on train and val sets
    if step % eval_interval == 0 or step == max_iters - 1:
        epoch_list.append(step)
        losses = estimate_loss()
        train_loss.append(losses['train'])
        val_loss.append(losses['val'])
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        with open('model_training_loss.txt', 'a') as f:
            f.write(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}\n")
        torch.save(m, 'model_backup.pth')

        
    x_train, y_train = get_random_batch('train', block_size = block_size, batch_size= batch_size)
    #print(x_train)
    logits, loss = m(x_train, y_train)
    optimizer.zero_grad(set_to_none = 1)
    loss.backward()
    optimizer.step()

import matplotlib.pyplot as plt
plt.figure()
plt.plot(epoch_list, train_loss)
plt.plot(epoch_list, val_loss)
plt.show()

print(decode(m.generate(idx =torch.zeros((1, 1), dtype=torch.long).to(device), max_new_tokens=1000)[0].tolist()))
```

### Saving the Model

```
torch.save(m, 'model.pth')
```

