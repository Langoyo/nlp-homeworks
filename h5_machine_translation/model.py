# BEGIN - DO NOT CHANGE THESE IMPORTS/CONSTANTS OR IMPORT ADDITIONAL PACKAGES.
import torch
import torch.nn as nn
import torch.nn.functional as F
# END - DO NOT CHANGE THESE IMPORTS/CONSTANTS OR IMPORT ADDITIONAL PACKAGES.

device = torch.device("cpu")#cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 10

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        Instantiate layers of encoder
        
        Model architecture will simply be embedding layer followed by GRU
        
        1. Initialize embedding layer with input_size different embeddings 
            and hidden_size as dimensions of each embedding
        2. input and hidden size of GRU will be hidden_size
        """
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
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
        
        Return: 
        - the new hidden state  (1 x 1 x hidden_size)
        - and output of the GRU (1 x 1 x hidden_size)
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
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        """
        Instantiate the layers for decoder
        
        The architecture is as follows:
        1. Embedding layer with output_size different embeddings
            and hidden_size as dimensions for each embedding
        2. GRU with input and hidden size as hidden_size
        3. Linear layer that takes the GRU's output
        4. Final softmax layer (use LogSoftmax) for linear
            layer's outputs
        """
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        ## YOUR CODE STARTS HERE (~4 lines of code) ##
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(2)   
        ## YOUR CODE ENDS HERE ##

    def forward(self, input, hidden):
        """
        Please implement the forward pass:
        1. Embed input same as with encoder.
        2. Pass them through ReLU (F.relu) before GRU
        3. Process output of 2 and hidden through GRU
        4. Execute linear layer
        5. Softmax linear layer's outputs
        
        Return: 
        - Output after softmax layer; shape: (output_size,) 
                you may need to squeeze() output
        - and new hidden returned by the GRU; shape: (1 x 1 x hidden_size)
        """
        output = None
        new_hidden = None
        ## YOUR CODE STARTS HERE (~5 lines of code) ##
        embeds = self.embedding(input)
        embeds = F.relu(embeds)
        embeds = torch.reshape(embeds,(1,1,-1))
        output, new_hidden = self.gru(embeds, hidden)
        output = self.linear(output)
        output = self.softmax(output)
        output = torch.squeeze(output)
        ## YOUR CODE ENDS HERE ##
        return output, new_hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

###### BONUS #########

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
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

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
