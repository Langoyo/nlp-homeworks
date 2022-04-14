# BEGIN - DO NOT CHANGE THESE IMPORTS OR IMPORT ADDITIONAL PACKAGES.
from unicodedata import bidirectional
import torch
import torch.nn as nn
# END - DO NOT CHANGE THESE IMPORTS OR IMPORT ADDITIONAL PACKAGES.
devicex = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ClassificationModel(nn.Module):
    # Instantiate layers for your model-
    # 
    # Your model architecture will be an optionally bidirectional LSTM,
    # followed by a linear + sigmoid layer.
    #
    # You'll need 4 nn.Modules
    # 1. An embeddings layer (see nn.Embedding)
    # 2. A bidirectional LSTM (see nn.LSTM)
    # 3. A Linear layer (see nn.Linear)
    # 4. A sigmoid output (see nn.Sigmoid)
    #
    # HINT: In the forward step, the BATCH_SIZE is the first dimension.
    # HINT: Think about what happens to the linear layer's hidden_dim size
    #       if bidirectional is True or False.
    # 
    def __init__(self, vocab_size, embedding_dim, hidden_dim, \
                 num_layers=1, bidirectional=True):
        super().__init__()
        ## YOUR CODE STARTS HERE (~4 lines of code) ##
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.embedded_layer = nn.Embedding(vocab_size,embedding_dim)
        if bidirectional:
            self.LSTM = nn.LSTM(embedding_dim,hidden_dim,num_layers=num_layers,bidirectional=bidirectional,batch_first=True)
            self.linear = nn.Linear(hidden_dim*2,3)
        else:
            self.LSTM = nn.LSTM(embedding_dim,hidden_dim,num_layers=num_layers,bidirectional=bidirectional,batch_first=True)
            self.linear = nn.Linear(hidden_dim,3)
        self.softmax = nn.Softmax(dim=1)
        ## YOUR CODE ENDS HERE ##
        
    # Complete the forward pass of the model.
    #
    # Use the last hidden timestep of the LSTM as input
    # to the linear layer. When completing the forward pass,
    # concatenate the last hidden timestep for both the foward,
    # and backward LSTMs.
    # 
    # args:
    # x - 2D LongTensor of shape (BATCH_SIZE, max len of all tokenized_word_tensor))
    #     This is the same output that comes out of the collate_fn function you completed-
    def forward(self, x):
        ## YOUR CODE STARTS HERE (~4-5 lines of code) ##
        
        #print(x)
        # print(x.get_device())
        x = self.embedded_layer(x)
        # x = torch.transpose(x,0,1)
        # Using only last hidden step
        out, (hn, cn) = self.LSTM(x)
        if bidirectional:
            hn = torch.cat([hn[0,:, :], hn[1,:,:]], dim=1).unsqueeze(0)
        x = self.linear(hn)
                
        return x
        ## YOUR CODE ENDS HERE ##
    
    # def init_weights(self):  # Randomly initilaize parameters
    #    initrange = 0.5
    #    self.embedded_layer.weight.data.uniform_(-initrange, initrange) # Uniform distribution
    #    self.linear.weight.data.uniform_(-initrange, initrange)
    #    self.linear.bias.data.zero_()  # Make bias zeros