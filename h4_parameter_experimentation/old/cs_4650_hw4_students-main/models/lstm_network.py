import torch.nn as nn
import torch


class LSTMNet(nn.Module):
    # define all the layers used in model
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 use_embeds,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional):
        """
        Define the parameters of your LSTM
        :param vocab_size: How many words should you expect?
        :param embedding_dim: How large should your embeddings be?
        :param use_embeds: Will you use external embeddings?
        :param hidden_dim: What will your hidden dimension be?
        :param output_dim: How many classes do you have?
        :param n_layers: How many layers should your LSTM be?
        :param bidirectional: Is the LSTM bi-directional?
        """
        # Constructor
        super(LSTMNet, self).__init__()
        # Create an embedding layer, an LSTM, and a Linear layer to predict classes per token
        ### YOUR CODE GOES HERE

        ### YOUR CODE ENDS HERE

    def forward(self, data_in):
        # If you're using pre-trained embeddings, skip your embedding layer
        # Otherwise, pass data_in through your embedding layer, then your LSTM, then your output layer.
        ### YOUR CODE GOES HERE

        ### YOUR CODE ENDS HERE
        return data_in


def loss(model, loss_fn, x, y, pre_trained, seq_len, device='cpu'):
    """
    Helper function to calculate loss for a set of samples
    :param model: model to use for the forward pass
    :param loss_fn: loss function to apply
    :param x: samples in
    :param y: labels out
    :param pre_trained: are we passing in embeddings or indices? Boolean: true for embeddings
    :param seq_len: sequence length
    :param device: device to put tensors on
    :return: torch loss value, accuracy number, and length of x (for reporting)
    """

    # If the length of x is not evenly divisible by your sequence length, need to fit the data somehow
    # Either remove trailing data, or (if x is too small), just set sequence-length to the length of x
    if len(x) % seq_len != 0:
        if len(x) < seq_len:
            seq_len = len(x)
        remainder = len(x) % seq_len
        x = x[:-remainder]
        y = y[:-remainder]

    # If we have embeddings going in, reshape to the batch x sequence x embed size
    # Else, just flatten into batches x seq_len (and cast to long)
    # cast labels to long and put everything on the right device
    # Pass through model, get loss with loss_fun, get accuracy from predictions,
    ### YOUR CODE GOES HERE

    ### YOUR CODE ENDS HERE
    loss_value = torch.tensor([0])  # replace this
    accuracy_value = 0  # replace this
    return loss_value, accuracy_value, len(x)


def train_loop(model, optimizer, loss_fn, samples, labels,
               batch_size, seq_len, device='cpu', pre_trained=False):
    """
    Standard pytorch training loop, using our helper loss function above.
    :param model: model to optimize
    :param optimizer: optimizer
    :param loss_fn: loss function
    :param samples: data in
    :param labels: labels out
    :param batch_size: batch size for sequences
    :param seq_len: sequence length
    :param device: device to put tensors on
    :param pre_trained: are we using pre-made embeddings or passing in indices?
    :return: model, loss, and accuracy
    """
    loss_total = 0
    acc_total = 0
    total_samples = 0
    # iterate through all samples, stepping by batch_size * sequence length and using
    # your loss function above to calculate loss. Then, zero gradients, backprop, step optimizer, and repeat
    # Also, store up the loss total, total number correct, and total number processed by the model so far

    for i in range(0, len(samples), batch_size*seq_len):
        ### YOUR CODE GOES HERE (delete the continue)
        continue
        ### YOUR CODE ENDS HERE
    # Return model, loss, and accuracy
    return model, loss_total, acc_total/total_samples


def val_loop(model, loss_fn, samples, labels, batch_size, seq_len, device='cpu', pre_trained=False):
    """
    Standard pytorch validation loop, using our helper loss function above
    :param model: model to test
    :param loss_fn: loss function to evaluate with
    :param samples: data in
    :param labels: labels out
    :param batch_size: batch size
    :param seq_len: sequence length
    :param device: device to put tensors on
    :param pre_trained: are we using pre-trained embeddings or indices?
    :return: loss and accuracy for evaluation
    """
    loss_total = 0
    acc_total = 0
    total_samples = 0
    with torch.no_grad():
        # Again, step through data taking batch_size*sequence_length sized steps
        # For each step, use your helper loss function to get a loss value and accuracy total
        # DO NOT STEP THE OPTIMIZER OR BACKPROP THE LOSS
        for i in range(0, len(samples), batch_size*seq_len):
            ### YOUR CODE GOES HERE (delete the continue)
            continue
            ### YOUR CODE ENDS HERE
    # Return loss and accuracy
    return loss_total, acc_total/total_samples
