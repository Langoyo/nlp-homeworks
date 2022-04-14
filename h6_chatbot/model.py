# BEGIN - DO NOT CHANGE THESE IMPORTS/CONSTANTS OR IMPORT ADDITIONAL PACKAGES.
import torch
import torch.nn as nn
import torch.nn.functional as F
# END - DO NOT CHANGE THESE IMPORTS/CONSTANTS OR IMPORT ADDITIONAL PACKAGES.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        """
        Instantiate layers of encoder
        
        Model architecture will simply be embedding layer followed by GRU
        
        1. Initialize embedding layer with input_size different embeddings 
            and hidden_size as dimensions of each embedding
        2. input and hidden size of GRU will be hidden_size
        """
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = n_layers
        ## YOUR CODE STARTS HERE (~2 lines of code) ##
        self.embedding = nn.Embedding(input_size,hidden_size)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size)
        ## YOUR CODE ENDS HERE ##

    def forward(self, input, hidden):
        """
        Implement forward pass of the model
        
        In our current implementation, we will manually call forward for each
            input sentence token. Therefore, input will only be one token. 
        
        For the forward pass:
        1. Embed input token with embedding layer (since sequence length and
            batch size is 1 you will need to reshape it to (1, 1, -1))
        2. Feed the embedded input and hidden state to GRU. 

        Inputs:
        - input: word_index
        - hidden: hidden tensor of shape (D * num_layers, 1, hidden_size)
            - Here D = 2 if bidirectional = True else 1 (D = 1 for us)
        Return: 
        - the new hidden state  (D * num_layers x 1 x hidden_size)
        - and output of the GRU (1 x 1 x hidden_size * D)
        """
        output = None
        new_hidden = None
        ## YOUR CODE STARTS HERE (~2 lines of code) ##
        embeds = self.embedding(input)
        embeds = torch.reshape(embeds,(1,1,-1))
        output,new_hidden = self.gru(embeds, hidden)
        ## YOUR CODE ENDS HERE ##
        return output, new_hidden

    def initHidden(self):
        D = 1
        return torch.zeros(D * self.num_layers, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    """
    The purpose of adding attention to the decoder is to allow it to focus on relevant
    parts of the encoder's outputs when generating its output. 

    Step 1
    --------
    We will first calculate attention score for to scale encoder outputs by relevance.
    For this part we need current input to the decoder and previous hidden state (for 
    the first time-step this is encoder's last hidden state). The process of obtaining 
    attention scores is as follows
    1. Embed decoder's input
    2. Pass it through a dropout layer for regularization purpose (you may skip this)
    3. Concatenate embedded input and previous hidden state
    4. Pass it through a linear layer to output max_length scores (this is 10 for us)
    the output of linear layer is the attention scores we will use in the next step

    Step 2
    -------
    Now we will scale encoder outputs by the attention and sum up the encoder outputs.
    You can use torch.bmm for this. 
    Given attention weights (1 x max_len) and encoder outputs (max_len x hidden_size), 
    after applying attention we should obtain vector of size (1 x hidden_size), which 
    should contain information about specific parts of the input that is relevant to 
    produce the word. 

    Step 3
    -------
    We will now combine the decoder's input and output of step 2 by using another linear
    layer to prepare a new input to the decoder (one that has knowledge of specific parts
    of the input sequence that are relevant). To do this:
    1. Concatenate decoder's input with output from step 2 
        (will yield in vector of size : hidden_size * 2)
    2. Pass it through a linear layer to output the new input of size hidden_size

    Step 4
    --------
    Pass the new input and previous hidden state through the GRU unit in the same manner 
    as our DecoderRNN 
    """
    def __init__(self, hidden_size, output_size, dropout_p=0.1, n_layers=1, max_length=30):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.num_layers = n_layers

        ## YOUR CODE STARTS HERE (~6 lines of code) ##
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        ## YOUR CODE ENDS HERE ##

    def forward(self, input, hidden, encoder_outputs):
        output = None
        new_hidden = None
        attn_weights = None # this time we will also output attention weights (for visualization later)
        ## YOUR CODE STARTS HERE (~10-12 lines of code) ##
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, new_hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        output = torch.squeeze(output)
        ## YOUR CODE ENDS HERE ##
        return output, new_hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
