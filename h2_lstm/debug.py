# %% [markdown]
# ### Part 0. Google Colab Setup

# %% [markdown]
# Hopefully you're looking at this notebook in Colab! 
# 1. First, make a copy of this notebook to your local drive, so you can edit it. 
# 2. Go ahead and upload the OnionOrNot.csv file from the [assignment zip](https://www.cc.gatech.edu/classes/AY2022/cs4650_fall/programming/h2_torch.zip) in the files panel on the left.
# 3. Right click in the files panel, and select 'Create New Folder' - call this folder src
# 4. Upload all the files in the src/ folder from the [assignment zip](https://www.cc.gatech.edu/classes/AY2022/cs4650_fall/programming/h2_torch.zip) to the src/ folder on colab.
# 
# ***NOTE: REMEMBER TO REGULARLY REDOWNLOAD ALL THE FILES IN SRC FROM COLAB.*** 
# 
# ***IF YOU EDIT THE FILES IN COLAB, AND YOU DO NOT REDOWNLOAD THEM, YOU WILL LOSE YOUR WORK!***
# 
# If you want GPU's, you can always change your instance type to GPU directly in Colab.

# %% [markdown]
# ### Part 1. Loading and Preprocessing Data [10 points]
# The following cell loads the OnionOrNot dataset, and tokenizes each data item

# %%
# DO NOT MODIFY #
import torch
import random
import numpy as np
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
import sklearn
# this is how we select a GPU if it's avalible on your computer.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
import pandas as pd
from src.preprocess import clean_text 
import nltk
from tqdm import tqdm

nltk.download('punkt')
df = pd.read_csv('train.csv', quotechar='"')
df["tokenized"] = df["text"].apply(lambda x: nltk.word_tokenize(clean_text(x.lower())))

# to convert authors into numbers
author_to_number = {
    'EAP': 0,
    'HPL': 1,
    'MWS': 2
    
}

# lowercase, removing punctuation and tookenize sentences. Converting labels to int
for i in range(len(df)):
    df['author'][i] = author_to_number[df['author'][i]]

# %% [markdown]
# Here's what the dataset looks like. You can index into specific rows with pandas, and try to guess some of these yourself :)

# %%
df.head()

# %%
df.iloc[42]

# %% [markdown]
# Now that we've loaded this dataset, we need to split the data into train, validation, and test sets. We also need to create a vocab map for words in our Onion dataset, which will map tokens to numbers. This will be useful later, since torch models can only use tensors of sequences of numbers as inputs. **Go to src/dataset.py, and fill out split_train_val_test, generate_vocab_map**

# %%
## TODO: complete these methods in src/dataset.py
from src.dataset import split_train_val_test, generate_vocab_map
df = df.sample(frac=1)

train_df, val_df, test_df = split_train_val_test(df, props=[.8, .1, .1])
train_vocab, reverse_vocab = generate_vocab_map(train_df)

# %%
# this line of code will help test your implementation
(len(train_df) / len(df)), (len(val_df) / len(df)), (len(test_df) / len(df))

# %% [markdown]
# PyTorch has custom Datset Classes that have very useful extentions. **Go to src/dataset.py, and fill out the HeadlineDataset class.** Refer to PyTorch documentation on Dataset Classes for help.

# %%
from src.dataset import HeadlineDataset
from torch.utils.data import RandomSampler
#print(train_df)

train_dataset = HeadlineDataset(train_vocab, train_df)
val_dataset = HeadlineDataset(train_vocab, val_df)
test_dataset = HeadlineDataset(train_vocab, test_df)

# Now that we're wrapping our dataframes in PyTorch datsets, we can make use of PyTorch Random Samplers.
train_sampler = RandomSampler(train_dataset)
val_sampler = RandomSampler(val_dataset)
test_sampler = RandomSampler(test_dataset)

# %% [markdown]
# We can now use PyTorch DataLoaders to batch our data for us. **Go to src/dataset.py, and fill out collate_fn.** Refer to PyTorch documentation on Dataloaders for help.

# %%
from torch.utils.data import DataLoader
from src.dataset import collate_fn
BATCH_SIZE = 16
train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, collate_fn=collate_fn)
val_iterator = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, collate_fn=collate_fn)
test_iterator = DataLoader(test_dataset, batch_size=BATCH_SIZE, sampler=test_sampler, collate_fn=collate_fn)


# %%
# # Use this to test your collate_fn implementation.

# # You can look at the shapes of x and y or put print 
# # statements in collate_fn while running this snippet

for x, y in test_iterator:
    print(x,y)
    break
test_iterator = DataLoader(test_dataset, batch_size=BATCH_SIZE, sampler=test_sampler, collate_fn=collate_fn)


# %% [markdown]
# ### Part 2: Modeling [10 pts]
# Let's move to modeling, now that we have dataset iterators that batch our data for us. **Go to src/model.py, and follow the instructions in the file to create a basic neural network. Then, create your model using the class, and define hyperparameters.** 

# %%
from src.models import ClassificationModel
model = None
### YOUR CODE GOES HERE (1 line of code) ###
model = ClassificationModel(len(train_vocab),embedding_dim=32,hidden_dim = 32,num_layers = 1,bidirectional = True)

# model.to(device)
# # 
### YOUR CODE ENDS HERE ###

# %% [markdown]
# In the following cell, **instantiate the model with some hyperparameters, and select an appropriate loss function and optimizer.** 
# 
# Hint: we already use sigmoid in our model. What loss functions are availible for binary classification? Feel free to look at PyTorch docs for help!

# %%
from torch.optim import AdamW

criterion, optimizer = None, None
### YOUR CODE GOES HERE ###
criterion, optimizer = torch.nn.CrossEntropyLoss(), torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

### YOUR CODE ENDS HERE ###

# %% [markdown]
# ### Part 3: Training and Evaluation [10 Points]
# The final part of this HW involves training the model, and evaluating it at each epoch. **Fill out the train and test loops below.**

# %%
# returns the total loss calculated from criterion
def train_loop(model, criterion, iterator):
    model.train()
    total_loss = 0
    
    for x, y in tqdm(iterator):
        optimizer.zero_grad()
        # x = x.to(device)
        # y = y.to(device)
        y = y.long()
        ### YOUR CODE STARTS HERE (~6 lines of code) ###
        prediction = model(x)
        prediction = torch.squeeze(prediction)

 
        loss = criterion(prediction,y)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    # scheduler.step()
        ### YOUR CODE ENDS HERE ###
    return total_loss

# returns:
# - true: a Python boolean array of all the ground truth values 
#         taken from the dataset iterator
# - pred: a Python boolean array of all model predictions. 
def val_loop(model, criterion, iterator):
    true, pred = [], []
    ### YOUR CODE STARTS HERE (~8 lines of code) ###
    for x, y in tqdm(iterator):
        # x = x.to(device)
        # y = y.to(device)
        # print("x",x)
        # print("y",y)  
    
        preds = model(x)
        preds = torch.squeeze(preds)
        for i_batch in range(len(y)):
            true.append(y[i_batch])
            pred.append(torch.argmax(preds[i_batch]))
            
            


    ### YOUR CODE ENDS HERE ###
    return true, pred


# %% [markdown]
# We also need evaluation metrics that tell us how well our model is doing on the validation set at each epoch. **Complete the functions in src/eval.py.**

# %%
from sklearn.metrics import f1_score, accuracy_score
# To test your eval implementation, let's see how well the untrained model does on our dev dataset.
# It should do pretty poorly.
from src.eval_utils import binary_macro_f1, accuracy
true, pred = val_loop(model, criterion, val_iterator)
true = [x.item() for x in true]
pred = [x.item() for x in pred]

print(f1_score(true, pred, average='weighted'))
print(accuracy_score(true, pred))


# %% [markdown]
# ### Part 4: Actually training the model [1 point]
# Watch your model train :D You should be able to achieve a validation F-1 score of at least .8 if everything went correctly. **Feel free to adjust the number of epochs to prevent overfitting or underfitting.**

# %%
TOTAL_EPOCHS = 7
for epoch in range(TOTAL_EPOCHS):
    train_loss = train_loop(model, criterion, train_iterator)
    true, pred = val_loop(model, criterion, val_iterator)
    print(f"EPOCH: {epoch}")
    print(f"TRAIN LOSS: {train_loss}")
    print(f"VAL F-1: {f1_score(true, pred, average='weighted')}")
    print(f"VAL ACC: {accuracy_score(true, pred)}")


# %% [markdown]
# We can also look at the models performance on the held-out test set, using the same val_loop we wrote earlier.

# %%
true, pred = val_loop(model, criterion, test_iterator)
print(f"TEST F-1: {f1_score(true, pred, average='weighted')}")
print(f"TEST ACC: {accuracy_score(true, pred)}")


