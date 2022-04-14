# %% [markdown]
# # HW5 Coding - Machine Translation
# 
# In this assignment, we will create a very simple Seq2Seq or encoder-decoder model for machine translation. A small dataset is constructed for this assignment for translating short sentences in french to english and can be found in the data folder. 
# 
# Before you start working with this homework, make sure to install the necessary dependencies with `pip install -r requirements.txt` (preferably in a virtual environment running python=3.7)

# %%

# %%
# Imports
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchtext.data.metrics import bleu_score
import pandas as pd
import time
import math
import numpy as np

# Plotting
# for colab
# %matplotlib inline 
# for local notebook

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%

# %%
from dataset import split_train_val, initLang, create_data_loader, SOS_token, EOS_token, UNK_token
from model import *

# %% [markdown]
# ## Read Data
# 
# Make sure `data/eng-fra.txt` file exists then read fra-eng translation data. Dataset is already cleaned and filtered to exclude non-letter characters.

# %%
data_df = pd.read_csv('data/fra-eng.csv')

# %% [markdown]
# Look at the data

# %%
print(data_df.shape)
data_df.sample(5)

# %% [markdown]
# ### Implement
# 
# We will now create create train / val splits. Please go to `dataset.py` and finish implementation for the function `split_train_val`

# %%
train_df, val_df = split_train_val(data_df, props=[0.8, 0.2])
# view train val size
print(train_df.shape, val_df.shape)
# verify implementation
assert train_df.shape[0] == 8479, "split is not implemented properly"
assert val_df.shape[0] == 2120, "split is not implemented properly"

# %% [markdown]
# Initialize vocabulary for input and target language vocabulary from train_df using the `initLang` function in `dataset.py`. This is already implemented for you.

# %%
[input_language_name, target_langauge_name] = train_df.columns.to_list()
print(f"Input language\t: {input_language_name}\nTarget language\t: {target_langauge_name}")

# %%
input_lang = initLang(name=input_language_name, df = train_df)
target_lang = initLang(name = target_langauge_name, df = train_df)

# %% [markdown]
# ### Implement
# 
# Create dataloaders for train, val and test data using `input_lang` and `target_lang` variables defined above. Please go to `dataset.py` and finish the implementation for `MT_Dataset.__getitem__()` in 

# %%
train_dataloader = create_data_loader(train_df, 
                                      input_lang=input_lang, 
                                      target_lang=target_lang,
                                      shuffle=True)
val_dataloader = create_data_loader(val_df, 
                                    input_lang=input_lang, 
                                    target_lang=target_lang, 
                                    shuffle=True)

# %%
# view sample from dataloader
next(iter(train_dataloader))

# %% [markdown]
# ## Model Seq2Seq
# 
# We will create a simle encoder-decoder machine translation model. Both encoder and decoders are RNNs. For this assigment, we will use gated recurrent units (GRU) for both encoder and decoder RNN. In the following section, you will implement the `EncoderRNN` and `DecoderRNN` models. The `EncoderRNN` will process tokens in input sentence one by one, where the last hidden state is used as context for the decoder. Ideally, this *context* should encode teh "meaning" of the input sequence for the decoder to efficiently produce the translated target sequence. The `DecoderRNN`'s first input will be a \<SOS\> token and context from the encoder. During test time, we would feed the decoder's output back in the next time step until it generates a \<EOS\> token or we reach some MAX_LEN. For training, we can randomly switch between feeding in decoder's own outputs as inputs and *teacher forcing* where we feed in the target tokens (true labels) regardless for what the decoder outputs at each time step. 
# 
# **Relevant work**:
# - [Sequence to Sequence Learnign with Neural Networks](https://arxiv.org/pdf/1409.3215.pdf)
# - [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation]

# %% [markdown]
# ### EncoderRNN and DecoderRNN
# 
# Please **implement** the `__init__` and `__forward__` functions of the `EncoderRNN` and `DecoderRNN` in `model.py`

# %% [markdown]
# ### Training the Model
# 
# Please **implement** the following `train` function which processes just one sample at a time. The train loop is implemented for you after this cell. The template initializes relevant variables. You will implement a for loop to step through encoder with `input_tensor` and a decoder for loop to step through `target_tensor` (with **and** without teacher forcing). The decoder loop is where you will also calculate the loss using the criterion. 
# 
# If you are unsure what arguments are being passed to the `train` function, please view the `train_iters` and `eval_iters` functions in the cells after. 

# %%
teacher_forcing_ratio = 0.5

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    # initialize encoder hidden state
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    loss = 0
    
    # Process input input_tensor (one token at a time with a for loop)
    ## YOUR CODE STARTS HERE (~2 lines of code) ##
    # for loop header
    decoder_input = []
    for sentence in input_tensor:
        # call encoder with input_tensor[ei] and encoder's hidden vector
        output, encoder_hidden = encoder(sentence, encoder_hidden)
        decoder_input.append(output)

    ## YOUR CODE ENDS HERE ##

    # Start decoder's input with <SOS> token
    decoder_input = torch.tensor([[SOS_token]], device=device)

    # initialize decoder hidden state as encoder's final hidden state
    decoder_hidden = encoder_hidden

    # decide whether to use teacher forcing 
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        ## YOUR CODE STARTS HERE (~4 lines of code) ##
        # for loop header to loop through target_tensor
        for word in target_tensor:
            # call decoder with decoder_input and decoder_hidden. make sure to reset
            #     decoder_hidden with the new hidden state returned
            
            output,decoder_hidden = decoder(decoder_input, decoder_hidden)

            # calculate loss using decoder's output and corresponding target_tensor[di]
            #     the criterion used is a negative log likelihood loss (NLLLoss)
            loss += criterion(output,word)

            # (Teacher forcing) set next decoder input as target_tensor[di]
            decoder_input = word

        ## YOUR CODE ENDS HERE ##

    else:
        # Without teacher forcing: use its own predictions as the next input
        ## YOUR CODE STARTS HERE (~5-7 lines of code) ##
        # for loop header to loop through target_tensor
        for di in range(target_length):
            # call decoder same as above
            output,decoder_hidden = decoder(decoder_input, decoder_hidden)
            # calculate loss same as above
            loss += criterion(output,target_tensor[di])
            # set next decoder input as argmax of decoder's output
            decoder_input = torch.argmax(output)
            # if new decoder_input is EOS_token: break
            if decoder_input == EOS_token:
                break
        ## YOUR CODE ENDS HERE ##

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

# %% [markdown]
# Next, please **implement** the `evaluate` function in the next cell. The process will be the same as `train`, except you will not use teacher forcing when stepping through the decoder. 

# %%
# evaluate
def evaluate(input_tensor, target_tensor, encoder, decoder, criterion, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_length = input_tensor.size()[0]
        target_length = target_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        loss = 0
        # Process input input_tensor (one token at a time with a for loop)
        ## YOUR CODE STARTS HERE (~2 lines of code) ##
        decoder_input = []
        for sentence in input_tensor:
        # call encoder with input_tensor[ei] and encoder's hidden vector
            output, encoder_hidden = encoder(sentence, encoder_hidden)
            decoder_input.append(output)
        ## YOUR CODE ENDS HERE ##

        # Start decoder's input with <SOS> token
        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        # initialize decoder hidden state as encoder's final hidden state
        decoder_hidden = encoder_hidden

        decoded_indices = []
        decoded_words = []

        # Run decoder starting with encoder's context (decoder_hidden)
        #     without using teacher forcing: use its own predictions as the next input
        ## YOUR CODE STARTS HERE (~5-7 lines of code) ##
        # for loop header to loop through target_tensor
        for di in range(target_length):
            # call decoder
            output,decoder_hidden = decoder(decoder_input, decoder_hidden)
            # calculate loss same as above
            loss += criterion(output,target_tensor[di])
            # set next decoder input as argmax of decoder's output
            decoder_input = torch.argmax(output)
            # append outputted index to decoded_indices
            decoded_indices.append(decoder_input)
            # if new decoder_input is EOS_token: break
            if decoder_input == EOS_token:
                break

        ## YOUR CODE ENDS HERE ##
        decoded_words = target_lang.getWordsFromIndices(decoded_indices)

        return decoded_words, loss.item() / len(decoded_words)

# %% [markdown]
# `trainEvalIters` is implemented for you. Simply review the implementation and run the following cell.

# %%
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def createDataFrame(points):
    df = pd.DataFrame(points, columns=['train_loss', 'val_loss', 'blue_scores_val'])
    return df

def evalIters(encoder, decoder, val_dataloader, criterion, num_samples=100):
    if num_samples > len(val_dataloader):
        num_samples = len(val_dataloader)
    
    candidates = [] # predicted output
    references = [] # true targets
    
    total_loss = 0
    for iter, sample in enumerate(val_dataloader):
        if iter > num_samples: break

        input_tensor = sample['input_tensor'].squeeze().to(device)
        target_tensor = sample['target_tensor'].squeeze().to(device)
        
        decoded_out, loss = evaluate(input_tensor, target_tensor, encoder, decoder, criterion)
        candidates.append(decoded_out)
        target_indices = sample['target_tensor'].squeeze().tolist()
        target_words = target_lang.getWordsFromIndices(target_indices)
        references.append([target_words])
        total_loss += loss
    
    return total_loss / num_samples, bleu_score(candidate_corpus=candidates, 
                                                references_corpus=references,
                                                max_n=4)

def trainEvalIters(encoder, decoder, epochs, train_dataloader, val_dataloader, 
                   eval_every=2000, eval_samples=200, learning_rate=0.01):
    start = time.time()
    plot_stats = []
    train_loss_total = 0  # Reset every eval_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()
    num_samples = len(train_dataloader)
    n_iters = epochs * num_samples
    
    for ep in range(epochs):
        for iter, sample in enumerate(train_dataloader):
            input_tensor = sample['input_tensor'].squeeze().to(device)
#             print(input_tensor.shape)
            target_tensor = sample['target_tensor'].squeeze().to(device)

            loss = train(input_tensor, target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion)
            train_loss_total += loss

            if (iter + 1) % eval_every == 0:
                train_loss_average = train_loss_total / eval_every
                train_loss_total = 0
                
                # validate model
                eval_loss, bleu_score = evalIters(encoder, decoder, val_dataloader, criterion, num_samples=eval_samples)
                
                curr_iter = ep * num_samples + iter + 1
                print('%s (%d %d%%) Average train loss: %.4f, Average val loss: %.4f ,val BLEU: %.4f' % (timeSince(start, curr_iter / n_iters), curr_iter, curr_iter / n_iters * 100, train_loss_average, eval_loss, bleu_score))

#                 plot_loss_avg = plot_loss_total / plot_every
                plot_stats.append([train_loss_average, eval_loss, bleu_score])
                plot_loss_total = 0

    return createDataFrame(plot_stats)

# %%
hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder1 = DecoderRNN(hidden_size, target_lang.n_words).to(device)

# training for 8 epochs on CPU will take about 25-30 mins (and ~15 mins with colab GPU)
epochs = 8

# losses_df = trainEvalIters(encoder1, decoder1, epochs, train_dataloader, val_dataloader, eval_every=2000, eval_samples=200, learning_rate=0.01)

# %%
# Plot losses
# losses_df.iloc[:, :-1].plot.line()

# %%
# losses_df.iloc[:, -1].plot.line()

# %% [markdown]
# ### View translations
# 
# Here is a quick function to manually assess the quality of your translator

# %%
def evaluateRandomly(encoder, decoder, val_dataloader, criterion, n=10):
    for i, sample in enumerate(val_dataloader):
        if i > n: break
        input_tensor = sample['input_tensor'].squeeze()
        target_tensor = sample['target_tensor'].squeeze()
        
        print('>', ' '.join(input_lang.getWordsFromIndices(input_tensor.tolist())))
        print('=', ' '.join(target_lang.getWordsFromIndices(target_tensor.tolist())))
        
        input_tensor = input_tensor.to(device)
        target_tensor = target_tensor.to(device)
        decoded_out, _ = evaluate(input_tensor, target_tensor, encoder, decoder, criterion)
        
        output_sentence = ' '.join(decoded_out)
        print('<', output_sentence)
        print('')

# %%
criterion = nn.NLLLoss()
# evaluateRandomly(encoder1, decoder1, val_dataloader, criterion)

# %% [markdown]
# ## Bonus add Attention
# 
# We can use the same encoder but create a modified version of `DecoderRNN` with attention applied. Implement `AttnDecoderRNN` in `model.py`. 

# %% [markdown]
# ### Training the Seq2Seq with Attention
# 
# 
# **TODO** modify instructions
# Please **implement** the following `train` and `evaluate` functions below. The overall structure is the same as before with a few differences. During encoder processing, we also need to keep track of all encoder outputs to feed into the decoder later. During decoder loop, we will have to readjust our call to add encoder outputs and receive decoder attention weights. 

# %%
teacher_forcing_ratio = 0.5
MAX_LENGTH = 10

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    # initialize encoder hidden state
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    loss = 0
    
    # this time we need to keep trak of encoder outputs
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    
    # Process input input_tensor (one token at a time with a for loop)
    ## YOUR CODE STARTS HERE (~3 lines of code) ##
    # for loop header
    for ei in range(input_length):
        # call encoder with input_tensor[ei] and encoder's hidden vector
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        # save encoder output in encoder_outputs
        encoder_outputs[ei] = encoder_output[0, 0]
    ## YOUR CODE ENDS HERE ##

    # Start decoder's input with <SOS> token
    decoder_input = torch.tensor([[SOS_token]], device=device)

    # initialize decoder hidden state as encoder's final hidden state
    decoder_hidden = encoder_hidden

    # decide whether to use teacher forcing 
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        ## YOUR CODE STARTS HERE (~4 lines of code) ##
        # for loop header to loop through target_tensor
        for di in range(target_length):
            # call decoder with decoder_input, decoder_hidden and encoder_outputs. 
            #     make sure to reset decoder_hidden with the new hidden state returned
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)

            # calculate loss using decoder's output and corresponding target_tensor[di]
            #     the criterion used is a negative log likelihood loss (NLLLoss)
            loss += criterion(decoder_output, target_tensor[di])

            # (Teacher forcing) set next decoder input as target_tensor[di]
            decoder_input = target_tensor[di] 

        ## YOUR CODE ENDS HERE ##

    else:
        # Without teacher forcing: use its own predictions as the next input
        ## YOUR CODE STARTS HERE (~5-7 lines of code) ##
        # for loop header to loop through target_tensor
        for di in range(target_length):
            
            # call decoder same as above
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            # calculate loss same as above
            loss += criterion(decoder_output, target_tensor[di])
            # set next decoder input as argmax of decoder's output

            # if new decoder_input is EOS_token: break
            if decoder_input.item() == EOS_token:
                break

        ## YOUR CODE ENDS HERE ##

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

# %% [markdown]
# Next, please **implement** the `evaluate` function in the next cell. The process will be the same as `train`, except you will not use teacher forcing when stepping through the decoder. 

# %%
# evaluate
def evaluate(input_tensor, target_tensor, encoder, decoder, criterion, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_length = input_tensor.size()[0]
        target_length = target_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        loss = 0
        
        # this time we need to keep trak of encoder outputs
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        
        # Process input input_tensor (one token at a time with a for loop)
        ## YOUR CODE STARTS HERE (~3 lines of code) ##
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        ## YOUR CODE ENDS HERE ##

        # Start decoder's input with <SOS> token
        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        # initialize decoder hidden state as encoder's final hidden state
        decoder_hidden = encoder_hidden

        decoded_indices = []
        decoded_words = []
        # keep track of decoder attention for analysis later
        decoder_attentions = torch.zeros(max_length, max_length)

        # Run decoder starting with encoder's context (decoder_hidden)
        #     without using teacher forcing: use its own predictions as the next input
        ## YOUR CODE STARTS HERE (~5-7 lines of code) ##
        # for loop header to loop through target_tensor
        for di in range(target_length):

            # call decoder
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            # save decoder attention in decoder_attentions
            decoder_attentions[di] = decoder_attention.data

            # calculate loss same as above
            loss += criterion(decoder_output, target_tensor[di])
            
            # set next decoder input as argmax of decoder's output
            topv, topi = decoder_output.data.topk(1)
            decoder_input = topi.squeeze().detach()
            # append outputted index to decoded_indices

            # if new decoder_input is EOS_token: break
            if topi.item() == EOS_token:
                decoded_indices.append(EOS_token)
                break
            else:
                decoded_indices.append(topi.item())

        ## YOUR CODE ENDS HERE ##
        decoded_words = target_lang.getWordsFromIndices(decoded_indices)

        return decoded_words, loss.item() / len(decoded_words), decoder_attentions[:di + 1]


# %% [markdown]
# Train and eval loops are implemented for you

# %%
def createDataFrame(points):
    df = pd.DataFrame(points, columns=['train_loss', 'val_loss', 'blue_scores_val'])
    return df

def evalIters(encoder, decoder, val_dataloader, criterion, num_samples=100):
    if num_samples > len(val_dataloader):
        num_samples = len(val_dataloader)
    
    candidates = [] # predicted output
    references = [] # true targets
    
    total_loss = 0
    for iter, sample in enumerate(val_dataloader):
        if iter > num_samples: break

        input_tensor = sample['input_tensor'].squeeze().to(device)
        target_tensor = sample['target_tensor'].squeeze().to(device)
        
        decoded_out, loss, _ = evaluate(input_tensor, target_tensor, encoder, decoder, criterion)
        candidates.append(decoded_out)
        target_indices = sample['target_tensor'].squeeze().tolist()
        target_words = target_lang.getWordsFromIndices(target_indices)
        references.append([target_words])
        total_loss += loss
    
    return total_loss / num_samples, bleu_score(candidate_corpus=candidates, 
                                                references_corpus=references,
                                                max_n=4)

def trainEvalIters(encoder, decoder, epochs, train_dataloader, val_dataloader, 
                   eval_every=2000, eval_samples=200, learning_rate=0.01):
    start = time.time()
    plot_stats = []
    train_loss_total = 0  # Reset every eval_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()
    num_samples = len(train_dataloader)
    n_iters = epochs * num_samples
    
    for ep in range(epochs):
        for iter, sample in enumerate(train_dataloader):
            input_tensor = sample['input_tensor'].squeeze().to(device)
#             print(input_tensor.shape)
            target_tensor = sample['target_tensor'].squeeze().to(device)

            loss = train(input_tensor, target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion)
            train_loss_total += loss

            if (iter + 1) % eval_every == 0:
                train_loss_average = train_loss_total / eval_every
                train_loss_total = 0
                
                # validate model
                eval_loss, bleu_score = evalIters(encoder, decoder, val_dataloader, criterion, num_samples=eval_samples)
                
                curr_iter = ep * num_samples + iter + 1
                print('%s (%d %d%%) Average train loss: %.4f, Average val loss: %.4f ,val BLEU: %.4f' % (timeSince(start, curr_iter / n_iters), curr_iter, curr_iter / n_iters * 100, train_loss_average, eval_loss, bleu_score))

#                 plot_loss_avg = plot_loss_total / plot_every
                plot_stats.append([train_loss_average, eval_loss, bleu_score])
                plot_loss_total = 0

    return createDataFrame(plot_stats)

# %%
hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder1 = AttnDecoderRNN(hidden_size, target_lang.n_words, dropout_p=0.1).to(device)

epochs = 8

losses_df = trainEvalIters(encoder1, decoder1, epochs, train_dataloader, val_dataloader, eval_every=2000, eval_samples=200, learning_rate=0.01)

# %%
# Plot losses
losses_df.iloc[:, :-1].plot.line()

# %%
losses_df.iloc[:, -1].plot.line()

# %% [markdown]
# ## Visualize Attention

# %%
import matplotlib.ticker as ticker
def showAttention(input_words, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_words, rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

def evaluateRandomlyAndShowAttention(encoder, decoder, val_dataloader, criterion, n = 10):

    for i, sample in enumerate(val_dataloader):
        if i > n: break
        input_words = input_lang.getWordsFromIndices(sample['input_tensor'].squeeze().tolist())
        input_tensor = sample['input_tensor'].squeeze().to(device)
        target_tensor = sample['target_tensor'].squeeze().to(device)
        output_words, _, attention = evaluate(input_tensor, target_tensor, encoder, decoder, criterion)
        target_words = target_lang.getWordsFromIndices(sample['target_tensor'].squeeze().tolist())
        print('input : ', ' '.join(input_words))
        print('target: ',' '.join(target_words))
        print('predicted :', ' '.join(output_words))
        showAttention(input_words, output_words, attention)

criterion = nn.NLLLoss()
evaluateRandomlyAndShowAttention(encoder1, decoder1, val_dataloader, criterion)

# %%


# def evaluateRandomly(encoder, decoder, val_dataloader, criterion, n=10):
#     for i, sample in enumerate(val_dataloader):
#         if i > n: break
#         input_tensor = sample['input_tensor'].squeeze()
#         target_tensor = sample['target_tensor'].squeeze()
        
#         print('>', ' '.join(input_lang.getWordsFromIndices(input_tensor.tolist())))
#         print('=', ' '.join(target_lang.getWordsFromIndices(target_tensor.tolist())))
        
#         input_tensor = input_tensor.to(device)
#         target_tensor = target_tensor.to(device)
#         decoded_out, _ = evaluate(input_tensor, target_tensor, encoder, decoder, criterion)
        
#         output_sentence = ' '.join(decoded_out)
#         print('<', output_sentence)
#         print('')