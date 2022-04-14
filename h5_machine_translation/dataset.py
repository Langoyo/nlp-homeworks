# BEGIN - DO NOT CHANGE THESE IMPORTS/CONSTANTS OR IMPORT ADDITIONAL PACKAGES.
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
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
def split_train_val(df: pd.DataFrame, props=[.8, .2]):
    assert round(sum(props), 2) == 1 and len(props) == 2
    # return values
    train_df, val_df = None, None

    ## YOUR CODE STARTS HERE (~6-10 lines of code)
    # hint: you can use df.iloc to slice into specific indexes
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

class Lang:
    def __init__(self, name, tokenizer=defaultTokenizer):
        """
        Initialize variables to maintain language statistics
        Args:
            name: str describing language (usually same as df column name)
            tokenizer: could be substituted according to specific language needs
                the default tokenizer will suffice for this assignment since the 
                data is already cleaned but if you use this template elsewhere, 
                you may want to substitute a different tokenizer
        """
        self.name = name
        self.word2index = {"UNK": UNK_token}
        # counts can be used to update dataset to use filter dataset samples by frequency
        self.word2count = {"UNK": 0}
        self.index2word = {SOS_token: "SOS",
                           EOS_token: "EOS", 
                           UNK_token: "UNK"}
        self.n_words = 3  # Count SOS, EOS, and UNK
        self.tokenizer = tokenizer

    def addSentence(self, sentence):
        for word in self.tokenizer(sentence):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
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
        if index in self.index2word:

            word = self.index2word[index]  
        else:
            word = "UNK"
        return word

    def getWordsFromIndices(self, indices):
        words = []
        for idx in indices:
            words.append(self.getWordFromIndex(idx))
        return words

def initLang(name, df, verbose=True):
    """
    Initialize Lang object, create word indices and update counts
    """
    lang = Lang(name)
    # Initialize vocabulary
    if verbose: 
        print(f'Counting words for lang={name} ...')
    df[name].apply(lang.addSentence)
    if verbose:
        print(f'{name} total word types:', lang.n_words)
    
    return lang

# Create Dataloader
class MT_Dataset(Dataset):
    def __init__(self, input_lang: Lang, target_lang: Lang, df: pd.DataFrame) -> None:
        super().__init__()
        self.input_lang = input_lang
        self.target_lang = target_lang
        self.df = df

    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, index: int):
        sample = self.df.iloc[index]
        input_sentence = sample[self.input_lang.name]
        target_sentence = sample[self.target_lang.name]
        input_tensor = None
        target_tensor = None
        # 1. Tokenize both sentences
        # 2. Map tokens to indices (don't forget to add EOS_token at the end) TODO add note about EOS token in notebook
        # 3. Convert list of indices to a torch tensor
        ## YOUR CODE STARTS HERE (~6-10 lines of code)
        input_tensor = defaultTokenizer(input_sentence)
        target_tensor = defaultTokenizer(target_sentence)
        for i in range(len(input_tensor)):
            input_tensor[i] = self.input_lang.getIndexFromWord(input_tensor[i])
        for i in range(len(target_tensor)):
            target_tensor[i] = self.target_lang.getIndexFromWord(target_tensor[i])
        input_tensor.append(EOS_token)
        target_tensor.append(EOS_token)
        input_tensor = torch.tensor(input_tensor)
        target_tensor = torch.tensor(target_tensor)
        ## YOUR CODE ENDS HERE ##

        return {
            'input_tensor' : input_tensor,
            'target_tensor' : target_tensor
        }

def create_data_loader(df, input_lang: Lang, target_lang: Lang, shuffle=True):
    ds = MT_Dataset(
        input_lang = input_lang,
        target_lang = target_lang,
        df = df
    )

    # batch_size 1 for this project
    return DataLoader(ds, batch_size=1, shuffle=shuffle)
