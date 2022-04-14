# BEGIN - DO NOT CHANGE THESE IMPORTS/CONSTANTS OR IMPORT ADDITIONAL PACKAGES.
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
# END - DO NOT CHANGE THESE IMPORTS/CONSTANTS OR IMPORT ADDITIONAL PACKAGES.


# split_train_val
# This method takes a dataframe and splits it into train/val splits.
# It uses the props argument to split the dataset appropriately.
#
# args:
# df - the entire dataset DataFrame
# props - proportions for each split
#
# returns:
# train DataFrame, val DataFrame
#
def split_train_val(df: pd.DataFrame, props=[.8, .2], shuffle=True):
    assert round(sum(props), 2) == 1 and len(props) == 2
    # return values
    train_df, val_df = None, None

    ## YOUR CODE STARTS HERE (~6-10 lines of code)
    # hint: you can use df.iloc to slice into specific indexes
    if shuffle:
        df.iloc[np.random.permutation(len(df))]
        df.reset_index(drop=True)
    length = len(df)
    train_df = df[:int(props[0]*length)]
    val_df = df[int(props[0]*length):]
    ## YOUR CODE ENDS HERE ##

    return train_df, val_df

# Default tokenizer that simply splits input by space
def defaultTokenizer(sentence):
    return sentence.split(' ')

# Special tokens
SOS_token = 0
EOS_token = 1
UNK_token = 2
SEP_token = 3

index_to_personas = {
    0:'ross',
    1:'rachel',
    2:'monica',
    3:'phoebe',
    4:'chandler',
    5:'other_char'
}
personas_to_index = {
    'ross':0,
    'rachel':1,
    'monica':2,
    'phoebe':3,
    'chandler':4,
    'other_char':5
}

class Lang:
    def __init__(self, tokenizer=defaultTokenizer):
        """
        Initialize variables to maintain language statistics
            Obtained from https://pytorch.org/tutorials/beginner/chatbot_tutorial.html#define-training-procedure
        Args:
            tokenizer: could be substituted according to specific language needs
                the default tokenizer will suffice for this assignment since the 
                data is already cleaned but if you use this template elsewhere, 
                you may want to substitute a different tokenizer
        """
        self.word2index = {"UNK": UNK_token,
                            "EOS" : EOS_token,
                            "UNK" : UNK_token,
                            "SEP" : SEP_token}
        # counts can be used to update dataset to use filter dataset samples by frequency
        self.word2count = {"UNK": 0, "SEP": 0}
        self.trimmed = False
        self.index2word = {SOS_token: "SOS",
                           EOS_token: "EOS", 
                           UNK_token: "UNK",
                           SEP_token: "SEP"}
        self.n_words = len(self.word2index)  # Count SOS, EOS, UNK, and SEP tokens
        self.tokenizer = tokenizer

    def addDialogue(self, tokenized_dialogue):
        # print(tokenized_dialogue)
        for word in tokenized_dialogue:
            # print(word)
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        elif word not in self.word2count:
            self.word2count[word] = 1
        else:
            self.word2count[word] += 1

    def getIndexFromWord(self, word):
        if word in self.word2index:
            return self.word2index[word]
        else:
            return self.word2index["UNK"]

    def getWordFromIndex(self, index):
        if not isinstance(index, int):
            index = index.item() 
        word = self.index2word[index] if index in self.index2word else "UNK"
        return word

    def getWordsFromIndices(self, indices):
        return [self.getWordFromIndex(idx) for idx in indices]
    
    def trim(self, min_count, keep=set()):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = list(keep)

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {"UNK": UNK_token,
                            "EOS" : EOS_token,
                            "UNK" : UNK_token,
                            "SEP" : SEP_token}
        self.word2count = {}
        self.index2word = {SOS_token: "SOS",
                           EOS_token: "EOS", 
                           UNK_token: "UNK",
                           SEP_token: "SEP"}
        self.n_words = len(self.word2index)  # Count SOS, EOS, UNK, and SEP tokens

        for word in keep_words:
            self.addWord(word)

def initLang(characters, df, verbose=True):
    """
    Initialize Lang object, create word indices and update counts
    """
    lang = Lang()

    # add characters to vocabulary
    for char in characters:
        lang.addWord(char)
    print('characters added | total_vocab_size:', len(lang.word2index))

    # Initialize vocabulary from dialogue
    cols = df.columns.tolist()
    for c in cols:
        df[c].apply(lang.addDialogue)
        print('initializing:', c, "| total_vocab_size:", len(lang.word2index))
    
    return lang

# Create Dataloader
class Chatbot_Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, lang: Lang, context_col:str, target_col:str, MAX_LEN=10) -> None:
        super().__init__()
        self.lang = lang
        self.df = df
        self.target_col = target_col
        self.input_col = context_col
        self.MAX_LEN = MAX_LEN
        self.character_col = 'character'
        assert self.input_col in self.df.columns
        assert self.target_col in self.df.columns

    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, index: int):
        sample = self.df.iloc[index]
        input_sentence = sample[self.input_col]
        target_sentence = sample[self.target_col]
        target_character = sample[self.character_col]
        input_tensor = []
        target_tensor = []
        persona_tensor = []
        # Sentence is already tokenized
        # 1. Make sure sequences are no longer than MAX_LEN-1 
        #    (because we will add EOS_token in the next step)
        #       for input_sentence: keep right-most MAX_LEN tokens
        #       for target_sentence: keep left-most MAX_LEN tokens
        # 2. Map tokens to indices (don't forget to add EOS_token at the end)
        # 3. Convert list of indices to a torch tensor
        # 4. Create a persona_tensor
        ## YOUR CODE STARTS HERE (~6-10 lines of code)
        if len(input_sentence) >= self.MAX_LEN:
            input_sentence = input_sentence[-(self.MAX_LEN-1):]
        if len(target_sentence) >= self.MAX_LEN:
            target_sentence = target_sentence[:self.MAX_LEN-1]
        ## YOUR CODE ENDS HERE ##
        for i in range(len(input_sentence)):
            input_tensor.append(self.lang.getIndexFromWord(input_sentence[i]))
        for i in range(len(target_sentence)):
            target_tensor.append(self.lang.getIndexFromWord(target_sentence[i]))
        input_tensor.append(EOS_token)
        target_tensor.append(EOS_token)
        input_tensor = torch.tensor(input_tensor)
        target_tensor = torch.tensor(target_tensor)
        # for i in range(len(target_character)):
        persona_tensor.append(personas_to_index[target_character])
        persona_tensor = torch.tensor(persona_tensor)
        return {
            'input_tensor' : input_tensor,
            'target_tensor' : target_tensor,
            'persona_tensor' : persona_tensor
        }

def create_data_loader(df, lang: Lang, context_col:str, 
                        target_col:str, MAX_LEN:int = 10, shuffle=True):

    ds = Chatbot_Dataset(
        df = df, 
        lang = lang, 
        context_col = context_col, 
        target_col = target_col,
        MAX_LEN = MAX_LEN
    )

    # batch_size 1 for this project
    return DataLoader(ds, batch_size=1, shuffle=shuffle)
