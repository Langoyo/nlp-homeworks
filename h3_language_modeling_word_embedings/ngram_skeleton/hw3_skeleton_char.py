import math, random

################################################################################
# Part 0: Utility Functions
################################################################################

def start_pad(n):
    ''' Returns a padding string of length n to append to the front of text
        as a pre-processing step to building n-grams '''
    return '~' * n

def ngrams(n, text):
    ''' Returns the ngrams of the text as tuples where the first element is
        the length-n context and the second is the character '''
    result = []
    if n == 0:
        for index in range(0,len(text),):
            result.append(('',text[index]))

    else:
        for index in range(0,len(text)):
            if index - n < 0:
                result.append((start_pad(n-index)+text[0:index],text[index]))
            else:
                result.append((text[index-n:index],text[index]))
    return result

def create_ngram_model(model_class, path, n=2, k=0):
    ''' Creates and returns a new n-gram model trained on the path file '''
    model = model_class(n, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        model.update(f.read())
    return model


################################################################################
# Part 1: Basic N-Gram Model
################################################################################

class NgramModel(object):
    ''' A basic n-gram model using add-k smoothing '''

    def __init__(self, n, k=0):
        self.n = n
        self.k = k
        self.vocabulary = []
        self.ngrams = []
        

    def get_vocab(self):
        ''' Returns the set of characters in the vocab '''
        return set(self.vocabulary)

    def update(self, text):
        ''' Updates the model n-grams based on text '''
        self.text = text
        self.ngrams = self.ngrams + ngrams(self.n, self.text)
        self.vocabulary = []
        for tuple in self.ngrams:
            if tuple[1] not in self.vocabulary:
                self.vocabulary.append(tuple[1])
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

    def prob(self, context, char):
        ''' Returns the probability of char appearing after context '''
        count_context = 0
        count_word = 0
        #MLE
        # for tuple in self.ngrams:
        #     if tuple[0] == context:
        #         if tuple[1] == char:
        #             count_word+=1
        #         count_context+=1
        # if count_context == 0:
        #     return 1 / len(self.vocabulary)
        # return count_word / count_context
        if self.k == 0:
            if context not in self.counts.keys():
                return 1 / len(self.vocabulary)
            if (context+char) not in self.counts.keys():
                return 0
            return self.counts[context+char]/(self.counts[context])    
        else:
            if context not in self.counts.keys() and (context+char) in self.counts.keys():
                return (self.counts[context+char]+self.k) / self.k*len(self.vocabulary)
            elif context in self.counts.keys() and (context+char) not in self.counts.keys():
                return self.k/(self.counts[context]+self.k*len(self.vocabulary))
            elif context not in self.counts.keys() and (context+char) not in self.counts.keys():
                return self.k/(self.k*len(self.vocabulary))
        
            return (self.counts[context+char]+self.k)/(self.counts[context]+self.k*len(self.vocabulary))



    def random_char(self, context):
        ''' Returns a random character based on the given context and the 
            n-grams learned by this model '''
        r = random.random()
        # Creating a sorted distribution of words by their probability
        # distrubution = []
        # for v in self.vocabulary:
        #     current_v_prob = self.prob(context,v)
        #     index_distribution = 0
        #     inserted = False
        #     while (index_distribution < len(distrubution) and inserted == False):
        #         if distrubution[index_distribution][0] > current_v_prob:
        #             distrubution[index_distribution:index_distribution] = [(current_v_prob,v)]
        #             inserted = True
        #         index_distribution+=1
        #     if index_distribution == len(distrubution) and inserted == False:
        #         distrubution.append((current_v_prob,v))
        
        # # Searching for the v with r probability
        # for tuple_index in range(len(distrubution)):
        #     if not distrubution[tuple_index][0] <= r:
        #         return distrubution[tuple_index-1][1]
        # return distrubution[0][1]
        acc = 0
        for char in self.vocabulary:
            prob = self.prob(context,char)
            acc+=prob
            if acc > r:
                return char
        return self.vocabulary[len(self.vocabulary)]


            

        

    def random_text(self, length):
        ''' Returns text of the specified character length based on the
            n-grams learned by this model '''
        context = start_pad(self.n)
        word = ''
        for i in range(length):
            new_char = self.random_char(context)        
            word = word + new_char
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
        addition = 0
        length = len(text)
        text = start_pad(self.n) + text
        for char in range(self.n,len(text)):
            prob = self.prob(text[char-self.n:char],text[char])
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
            
        #     self.vocabulary = []
        #     for tuple in self.ngrams:
        #         if tuple[1] not in self.vocabulary:
        #             self.vocabulary.append(tuple[1])
        
        # self.vocabulary.sort()
        # self.counts = {}
        # for tuple in self.ngrams:
        #     if tuple[0] not in self.counts.keys():
        #         self.counts[tuple[0]]=1
        #     else: 
        #         self.counts[tuple[0]]+=1
        #     if (tuple[0]+tuple[1]) not in self.counts.keys():
        #         self.counts[tuple[0]+tuple[1]]=1
        #     else:
        #         self.counts[tuple[0]+tuple[1]]+=1
        # return self.ngrams

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