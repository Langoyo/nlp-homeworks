# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# 
# 
# # Generating Shakespeare with a Character-Level RNN
# 
# %% [markdown]
# In this part, we'll turn from traditional n-gram based language models to a more advanced form of language modeling using a Recurrent Neural Network. Specifically, we'll be setting up a character-level recurrent neural network (char-rnn) for short.
# 
# Andrej Karpathy, a researcher at OpenAI, has written an excellent blog post about using RNNs for language models, which you should read before beginning this assignment.  The title of his blog post is [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).
# 
# Karpathy shows how char-rnns can be used to generate texts for several fun domains:
# * Shakespeare plays
# * Essays about economics
# * LaTeX documents
# * Linux source code
# * Baby names
# 
# # Recommended Reading
# 
# You should install PyTorch, know Python, and understand Tensors:
# 
# * http://pytorch.org/ For installation instructions
# * [Deep Learning with PyTorch: A 60-minute Blitz](https://github.com/pytorch/tutorials/blob/master/Deep%20Learning%20with%20PyTorch.ipynb) to get started with PyTorch in general
# * [jcjohnson's PyTorch examples](https://github.com/jcjohnson/pytorch-examples) for an in depth overview
# * [Introduction to PyTorch for former Torchies](https://github.com/pytorch/tutorials/blob/master/Introduction%20to%20PyTorch%20for%20former%20Torchies.ipynb) if you are former Lua Torch user
# 
# It would also be useful to know about RNNs and how they work:
# 
# * [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) shows a bunch of real life examples
# * [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) is about LSTMs specifically but also informative about RNNs in general
# 
# Also see these related tutorials from the series:
# 
# * [Classifying Names with a Character-Level RNN](https://github.com/spro/practical-pytorch/blob/master/char-rnn-classification/char-rnn-classification.ipynb) uses an RNN for classification
# * [Generating Names with a Conditional Character-Level RNN](https://github.com/spro/practical-pytorch/blob/master/conditional-char-rnn/conditional-char-rnn.ipynb) builds on this model to add a category as input
# 
# ## You can also set up Pytorch in Google Colab
# 
# Pytorch is one of the most popular deep learning frameworks in both industry and academia, and learning its use will be invaluable should you choose a career in deep learning. 
# 
# ### Setup
# #### Using Google Colab (recommended)
# 1. Upload this notebook on [Colab](https://colab.research.google.com/notebooks/welcome.ipynb).
# 2. Set hardware accelerator to ```GPU``` under ```notebook settings``` in the ```Edit``` menu.
# 3. Run the first cell to  set up  the environment.
# 
# ### Note
# Please look at the FAQ section before you start working.
# 
# %% [markdown]
# # Prepare data
# 
# The file we are using is a plain text file. We turn any potential unicode characters into plain ASCII by using the `unidecode` package (which you can install via `pip` or `conda`).

# %%
import unidecode
import string
import random
import re
import torch

all_characters = string.printable
n_characters = len(all_characters)

file = unidecode.unidecode(open('shakespeare_data/shakespeare_input.txt').read())
file_len = len(file)
print('file_len =', file_len)

# %% [markdown]
# To make inputs out of this big string of data, we will be splitting it into chunks.

# %%
chunk_len = 200

def random_chunk():
    start_index = random.randint(0, file_len - chunk_len)
    end_index = start_index + chunk_len + 1
    return file[start_index:end_index]

print(random_chunk())

# %% [markdown]
# # Build the Model
# 
# This model will take as input the character for step $t_{-1}$ and is expected to output the next character $t$. There are three layers - one linear layer that encodes the input character into an internal state, one GRU layer (which may itself have multiple layers) that operates on that internal state and a hidden state, and a decoder layer that outputs the probability distribution. You need to finish the forward method. (Refer to [Pytorch GRU Documentation](https://pytorch.org/docs/stable/nn.html#gru))

# %%
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, hidden):
        #Input input: torch Tensor of shape (1,)
        #hidden: torch Tensor of shape (self.n_layers, 1, self.hidden_size)
        #Return output: torch Tensor of shape (1, self.output_size) 
        #and hidden: torch Tensor of shape (self.n_layers, 1, self.hidden_size)
        encoded = self.encoder(input)
        # encoded = encoded.squeeze(0)
        # print(encoded.view())
        # print(hidden.view())
        encoded = encoded.unsqueeze(0)
        encoded = encoded.unsqueeze(0)
        output, hidden = self.gru(encoded,hidden)
        # hidden = hidden.view(n_layers, n_directions=1, batch_size=1, hidden_dim=self.hidden_size)
        # hidden = hidden[-1]
        # output = output.squeeze(1)
        # hidden_forward, hidden_backward = hidden[0], hidden[1]
        out = self.decoder(output)
        out = out.squeeze(1)
        return out, hidden
        

    def init_hidden(self):
        return torch.zeros(self.n_layers, 1, self.hidden_size).to(device)

# %% [markdown]
# # Inputs and Targets
# %% [markdown]
# Each chunk will be turned into a tensor, specifically a `LongTensor` (used for integer values), by looping through the characters of the string and looking up the index of each character in `all_characters`.

# %%
# Turn string into list of longs
def char_tensor(string):
    tensor = torch.zeros(len(string)).long().to(device)
    for c in range(len(string)):
        tensor[c] = all_characters.index(string[c])
    return tensor

print(char_tensor('abcDEF'))

# %% [markdown]
# Finally we can assemble a pair of input and target tensors for training, from a random chunk. The input will be all characters *up to the last*, and the target will be all characters *from the first*. So if our chunk is "abc" the input will correspond to "ab" while the target is "bc".

# %%
def random_training_set():    
    chunk = random_chunk()
    inp = char_tensor(chunk[:-1])
    target = char_tensor(chunk[1:])
    return inp, target

# %% [markdown]
# # Evaluating
# 
# To evaluate the network we will feed one character at a time, use the outputs of the network as a probability distribution for the next character, and repeat. To start generation we pass a priming string to start building up the hidden state, from which we then generate one character at a time.

# %%
def evaluate(prime_str='A', predict_len=100, temperature=0.8):
    hidden = decoder.init_hidden()
    prime_input = char_tensor(prime_str)
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[p], hidden)
    inp = prime_input[-1]
    
    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)
        
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        
        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = char_tensor(predicted_char)[0]

    return predicted

# %% [markdown]
# # Training
# %% [markdown]
# A helper to print the amount of time passed:

# %%
import time, math

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# %% [markdown]
# The main training function

# %%
def train(inp, target):
    hidden = decoder.init_hidden()
    decoder.zero_grad()
    loss = 0

    for c in range(chunk_len):
        output, hidden = decoder(inp[c], hidden)
        loss += criterion(output, target.unsqueeze(1)[c])

    loss.backward()
    decoder_optimizer.step()

    return loss.item() / chunk_len

# %% [markdown]
# Then we define the training parameters, instantiate the model, and start training:

# %%
n_epochs = 2000
print_every = 100
plot_every = 10
hidden_size = 100
n_layers = 1
lr = 0.005

decoder = RNN(n_characters, hidden_size, n_characters, n_layers).to(device)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

start = time.time()
all_losses = []
loss_avg = 0

for epoch in range(1, n_epochs + 1):
    loss = train(*random_training_set())       
    loss_avg += loss

    if epoch % print_every == 0:
        print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / n_epochs * 100, loss))
        print(evaluate('Wh', 100), '\n')

    if epoch % plot_every == 0:
        all_losses.append(loss_avg / plot_every)
        loss_avg = 0

# %% [markdown]
# # Plotting the Training Losses
# 
# Plotting the historical loss from all_losses shows the network learning:

# %%
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure()
plt.plot(all_losses)

# %% [markdown]
# # Evaluating at different "temperatures"
# 
# In the `evaluate` function above, every time a prediction is made the outputs are divided by the "temperature" argument passed. Using a higher number makes all actions more equally likely, and thus gives us "more random" outputs. Using a lower value (less than 1) makes high probabilities contribute more. As we turn the temperature towards zero we are choosing only the most likely outputs.
# 
# We can see the effects of this by adjusting the `temperature` argument.

# %%
print(evaluate('Th', 200, temperature=0.8))

# %% [markdown]
# Lower temperatures are less varied, choosing only the more probable outputs:

# %%
print(evaluate('Th', 200, temperature=0.2))

# %% [markdown]
# Higher temperatures more varied, choosing less probable outputs:

# %%
print(evaluate('Th', 200, temperature=1.4))


# %%
import torch.nn.functional as F
def perp(testfile):
    inp = char_tensor(testfile[:-1])
    target = char_tensor(testfile[1:])
    test_len=len(testfile)
    hidden = decoder.init_hidden()
    decoder.zero_grad()
    perplexity=torch.tensor(0.0)

    for c in range(test_len-1):
        output, hidden = decoder(inp[c], hidden)
        perplexity -=F.log_softmax(output,dim=1)[0][target[c]]

    return (perplexity/test_len).exp().item()

testfile = unidecode.unidecode(open('data/shakespeare_sonnets.txt').read())
print('Perplexity:',perp(testfile))

# %% [markdown]
# ## FAQs
# 
# #### I'm unfamiliar with PyTorch. How do I get started?
# If you are new to the paradigm of computational graphs and functional programming, please have a look at this [tutorial](https://hackernoon.com/linear-regression-in-x-minutes-using-pytorch-8eec49f6a0e2) before getting started.
# 
# #### How do I speed up training?
# Send the model and the input, output tensors to the GPU using ```.to(device)```. Refer the [PyTorch docs](https://pytorch.org/docs/stable/notes/cuda.html) for further information.
# 

