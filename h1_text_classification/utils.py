from nltk.tokenize import regexp_tokenize
import numpy as np

# Here is a default pattern for tokenization, you can substitue it with yours
default_pattern =  r"""(?x)                  
                        (?:[A-Z]\.)+          
                        |\$?\d+(?:\.\d+)?%?    
                        |\w+(?:[-']\w+)*      
                        |\.\.\.               
                        |(?:[.,;"'?():-_`])    
                    """

def tokenize(text, pattern = default_pattern):
    """Tokenize senten with specific pattern
    
    Arguments:
        text {str} -- sentence to be tokenized, such as "I love NLP"
    
    Keyword Arguments:
        pattern {str} -- reg-expression pattern for tokenizer (default: {default_pattern})
    
    Returns:
        list -- list of tokenized words, such as ['I', 'love', 'nlp']
    """
    text = text.lower()
    return regexp_tokenize(text, pattern)


class FeatureExtractor(object):
    """Base class for feature extraction.
    """
    def __init__(self):
        pass
    def fit(self, text_set):
        pass
    def transform(self, text):
        pass  
    def transform_list(self, text_set):
        pass




class UnigramFeature(FeatureExtractor):
    """Example code for unigram feature extraction
    """
    def __init__(self):
        # dictionary that maps unigrams into their indexes by order of appearance
        self.unigram = {}
        
    def fit(self, text_set: list):
        """Fit a feature extractor based on given data 
        
        Arguments:
            text_set {list} -- list of tokenized sentences and words are lowercased, such as [["I", "love", "nlp"], ["I", "like", "python"]]
        """
        index = 0
        for i in range(0, len(text_set)):
            for j in range(0, len(text_set[i])):
                if text_set[i][j].lower() not in self.unigram:
                    self.unigram[text_set[i][j].lower()] = index
                    index += 1
                else:
                    continue
                    
    def transform(self, text: list):
        """Transform a given sentence into vectors based on the extractor you got from self.fit()
        
        Arguments:
            text {list} -- a tokenized sentence (list of words), such as ["I", "love", "nlp"]
        
        Returns:
            array -- an unigram feature array, such as array([1,1,1,0,0,0])
        """
        # Feature has the size of the number of distinct words received
        feature = np.zeros(len(self.unigram))
        # We go through every word in the text
        for i in range(0, len(text)):
            if text[i].lower() in self.unigram:
                # If it contained in the unigram,
                # It is a feature of the new text.
                # We search for the index of the word in the text and use that index in the feature as well
                # Then, we increase the number of occurrences of that word.
                # Seems like the bag of words seen in class.
                feature[self.unigram[text[i].lower()]] += 1
        
        return feature
    
    
    def transform_list(self, text_set: list):
        """Transform a list of tokenized sentences into vectors based on the extractor you got from self.fit()
        
        Arguments:
            text_set {list} --a list of tokenized sentences, such as [["I", "love", "nlp"], ["I", "like", "python"]]
        
        Returns:
            array -- unigram feature arraies, such as array([[1,1,1,0,0], [1,0,0,1,1]])
        """
        # Same as previous method but at sentences level
        features = []
        for i in range(0, len(text_set)):
            features.append(self.transform(text_set[i]))
        
        
        
        return np.array(features)

    def indexesToWords(self, words):
        for word, index in self.unigram.items():
            if index in words:
                print(word)
        
    


class BigramFeature(FeatureExtractor):
    """Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self):
        self.bigram = {}
    def fit(self, text_set):
        index = 0
        for i in range(0, len(text_set)):
            for j in range(0, len(text_set[i])-1):
                merged_words = (text_set[i][j]+text_set[i][j+1]).lower()
                if merged_words not in self.bigram:
                    self.bigram[merged_words] = index
                    index += 1
                else:
                    continue

    def transform(self, text):
        # Feature has the size of the number of distinct words received
        feature = np.zeros(len(self.bigram))
        # We go through every word in the text
        for i in range(0, len(text)-1):
            merged_words = (text[i].lower()+text[i+1].lower())
            if merged_words in self.bigram:
                # If it contained in the unigram,
                # It is a feature of the new text.
                # We search for the index of the word in the text and use that index in the feature as well
                # Then, we increase the number of occurrences of that word.
                # Seems like the bag of words seen in class.
                feature[self.bigram[merged_words]] += 1
        
        return feature
    def transform_list(self, text_set):
        # Add your code here!
        features = []
        for i in range(0, len(text_set)):
            features.append(self.transform(text_set[i]))
        
        return np.array(features)

    def indexesToWords(self, words):
        for word, index in self.unigram.items():
            if index in words:
                print(word)

    


class CustomFeature(FeatureExtractor):
    """customized feature extractor,nTF-IDF (term frequency / inverse document frequency)
    """
    def __init__(self):
        # Add your code here!
        self.unigram = {}


    def fit(self, text_set):
        # Creates unigrams dict
        index = 0
        for i in range(0, len(text_set)):
            for j in range(0, len(text_set[i])):
                if text_set[i][j].lower() not in self.unigram:
                    self.unigram[text_set[i][j].lower()] = index
                    index += 1
                else:
                    continue
                
    def transform(self, text):
        # obtainst the term freq of every word
        tf = np.zeros(len(self.unigram))
        # We go through every word in the text
        for i in range(0, len(text)):
            if text[i].lower() in self.unigram:
                
                tf[self.unigram[text[i].lower()]] += 1
        tf = tf/ len(text)
        return tf
        
    def transform_list(self, text_set):
        # Obtains tf-idf

        # Obtaining the tf of every document-word
        tf = []
        for i in range(0, len(text_set)):
            tf.append(self.transform(text_set[i]))

        # Getting the document freq of every word
        df = np.zeros(len(self.unigram))
        for document in range(len(text_set)):
            for word in range(len(self.unigram)):            
                # knowing in how many documents the word appear
                if tf[document][word]>0:
                    df[word]+=1

        # Calculating the inverse document freq of a word
        idf = np.zeros(len(self.unigram))
        for word in range(len(self.unigram)):
                idf[word] = np.log(len(text_set)/(df[word]+1))

        tf_idf = []
        
        for document in range(len(text_set)):
            tmp = []
            for word in range(len(self.unigram)):
                tmp.append(tf[document][word]*idf[word])
            tf_idf.append(tmp)

        
        
        return np.array(tf_idf)



        
