import math, random
from typing import List, Tuple

################################################################################
# Part 0: Utility Functions
################################################################################

def start_pad(n):
    ''' Returns a padding string of length n to append to the front of text
        as a pre-processing step to building n-grams '''
    return ['~'] * n

Pair = Tuple[str, str]
Ngrams = List[Pair]
def ngrams(n, text:str) -> Ngrams:
    text = text.replace('[EOS]','')
    text=text.strip().split()
    ''' Returns the ngrams of the text as tuples where the first element is
        the n-word sequence (i.e. "I love machine") context and the second is the word '''
    result = []
    if n == 0:
        for index in range(0,len(text),):
            result.append(('',text[index]))

    else:
        for index in range(0,len(text)):
            if index - n < 0:
                result.append((' '.join(start_pad(n-index)+text[0:index]),text[index]))
            else:
                result.append((' '.join(text[index-n:index]),text[index]))
    return result

def create_ngram_model(model_class, path, n=2, k=0):
    ''' Creates and returns a new n-gram model trained on the path file '''
    model = model_class(n, k)
    with open(path, encoding='utf-8') as f:
        model.update(f.read())
    return model

################################################################################
# Part 1: Basic N-Gram Model
################################################################################

class NgramModel(object):
    ''' A basic n-gram model using add-k smoothing '''

    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.count = {}
        self.ngrams = []

    def get_vocab(self):
        ''' Returns the set of words in the vocab '''
        return set(self.vocabulary)

    def update(self, text:str):
        ''' Updates the model n-grams based on text '''
        # text = text.replace('[EOS]','')
        if '[EOF]' in text:
            text = text.split('[EOS]')
            for sentence in text:
                self.ngrams = self.ngrams + ngrams(self.n, sentence)
        else:   
            self.ngrams = self.ngrams + ngrams(self.n, text)
        self.vocabulary = []
        added = {}
        for tuple in self.ngrams:
            if tuple[1] not in added.keys():
                self.vocabulary.append(tuple[1])
                added[tuple[1]]=1
        self.vocabulary.sort()
        self.counts = {}
        for tuple in self.ngrams:
            if tuple[0] not in self.counts.keys():
                self.counts[tuple[0]]=1
            else: 
                self.counts[tuple[0]]+=1
            if (tuple[0]+tuple[1]) not in self.counts.keys():
                self.counts[tuple[0]+tuple[1]]=1
            else:
                self.counts[tuple[0]+tuple[1]]+=1
        return self.ngrams

    def prob(self, context:str, word:str):
        ''' Returns the probability of word appearing after context '''
        if self.k == 0:
            if context not in self.counts.keys():
                return 1 / len(self.vocabulary)
            if (context+word) not in self.counts.keys():
                return 0
            return self.counts[context+word]/(self.counts[context])    
        else:
            if context not in self.counts.keys() and (context+word) in self.counts.keys():
                return (self.counts[context+word]+self.k) / self.k*len(self.vocabulary)
            elif context in self.counts.keys() and (context+word) not in self.counts.keys():
                return self.k/(self.counts[context]+self.k*len(self.vocabulary))
            elif context not in self.counts.keys() and (context+word) not in self.counts.keys():
                return self.k/(self.k*len(self.vocabulary))
        
            return (self.counts[context+word]+self.k)/(self.counts[context]+self.k*len(self.vocabulary))

    def random_word(self, context):
        ''' Returns a random word based on the given context and the
            n-grams learned by this model '''
        random.seed(1)
        r = random.random()
        acc = 0
        for char in self.vocabulary:
            prob = self.prob(context,char)
            acc+=prob
            if acc > r:
                return char
        return self.vocabulary[len(self.vocabulary)]

    def random_text(self, length):
        ''' Returns text of the specified word length based on the
            n-grams learned by this model '''
        context = ''.join(start_pad(self.n))
        word = ''
        for i in range(length):
            new_char = self.random_word(context)
            if i == 0:
                word = word + new_char
            else:       
                word = word + ' ' + new_char
            if self.n > 0:
                if self.n > 1:
                    context = context + new_char
                    if len(context)>self.n:
                        context = context[1:]
                else:
                    context = new_char
        return word

    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''
        if '[EOF]' in text:
            text = text.split('[EOS]')
        

            addition = 0
            length = len(text)
            text = start_pad(self.n) + text.strip().split()
            for sentence in text:
                for char in range(self.n,len(sentence)):
                    prob = self.prob(''.join(sentence[char-self.n:char]),sentence[char])
                    if prob == 0:
                        return float('inf')
                    addition += math.log(prob, 2)
            l = addition / length
        
        else:
            addition = 0
            length = len(text)
            text = start_pad(self.n) + text.strip().split()
            for char in range(self.n,len(text)):
                prob = self.prob(''.join(text[char-self.n:char]),text[char])
                if prob == 0:
                    return float('inf')
                addition += math.log(prob, 2)
            l = addition / length
        return math.pow(2,-l)



################################################################################
# Part 2: N-Gram Model with Interpolation
################################################################################

class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''

    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.vocabulary = []
        self.ngrams = []
        self.lambdas = []
        self.models = []
        for i in range(n+1):
            self.lambdas.append(1/(n+1))
        for ns in range(0,self.n+1):
            newModel = NgramModel(ns,self.k)
            self.models.append(newModel)
    def set_lambdas(self,lambdas):
        
        self.lambdas = lambdas

    def get_vocab(self):
        result = []
        for model in self.models:
            result = result + model.vocabulary
        return result

    def update(self, text):
        self.text = text
        for i in range(0,len(self.models)):        
            self.models[i].update(text)

    def prob(self, context, char):
        result = 0
        for i in range(0,self.n+1):
            result += self.lambdas[i] * self.models[i].prob(context[self.n-i:],char)
        return result

################################################################################
# Part 3: Your N-Gram Model Experimentation
################################################################################

if __name__ == '__main__':
    pass