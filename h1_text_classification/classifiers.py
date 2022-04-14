import numpy as np
import utils
# You need to build your own model here instead of using well-built python packages such as sklearn

#from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import Perceptron
# You can use the models form sklearn packages to check the performance of your own models

class HateSpeechClassifier(object):
    """Base class for classifiers.
    """
    def __init__(self):
        pass
    def fit(self, X, Y):
        """Train your model based on training set
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
            Y {type} -- array of actual labels, such as an N shape array, where N is the nu,ber of sentences
        """
        pass
    def predict(self, X):
        """Predict labels based on your trained model
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
        
        Returns:
            array -- predict labels, such as an N shape array, where N is the nu,ber of sentences
        """
        pass


class AlwaysPreditZero(HateSpeechClassifier):
    """Always predict the 0
    """
    def predict(self, X):
        return [0]*len(X)


class NaiveBayesClassifier(HateSpeechClassifier):
    """Naive Bayes Classifier
    """
    def __init__(self):
        # Add your code here!
        pass        

    def fit(self, X, Y):

        #self.clf = MultinomialNB()
        #self.clf.fit(X, Y)
        
        # Add your code here!
        # We create an array that will contain the occurrences of the words for each class
        self.table = np.zeros((2,X[0].size))
        # Number of words for each class
        self.words_in_a_class = np.zeros(2)
        # Number of documents classified for each class
        self.n_documents_class = np.zeros(2)
        # Number of documents
        self.n_documents = Y.size
        # Size of the vocabulary
        self.vocab = X[0].size
        # We iterate through every word in every sentence
        
        # Iterate over every document
        for i in range(Y.size):
            # iterate over each word in the document
            for j in range (len(X[0])):
                # We save the occurrences of every word depending if the class is 0 or 1
                if Y[i]==1 :
                    # Updating the frequency of the current word in the class 1
                    self.table[1][j] += X[i][j]
                    # Accumulating the number of words in class 1
                    self.words_in_a_class[1] += X[i][j]
        
                else:
                    # Updating the frequency of the current word in the class 0
                    self.table[0][j] += X[i][j]
                    # Accumulating the number of words in class 0
                    self.words_in_a_class[0] += X[i][j]
            # Note to which class this document belongs        
            if Y[i]==1:
                self.n_documents_class[1] += 1
            else:
                self.n_documents_class[0] += 1

            

    
    def predict(self, X):
        # Add your code here!
        
        predictions = [] 
        #predictions = self.clf.predict(X)
        
        # Go over every document
        for i in range(X.shape[0]):
            # Calculate the probability of each type of document
            negative_pred = np.log(self.n_documents_class[0]/self.n_documents)
            positive_pred = np.log(self.n_documents_class[1]/self.n_documents)
            for j in range(X[0].size):                
                if X[i][j] > 0:
                    # Adding the logs probabilities of the words found in the document
                    negative_pred += np.log( (self.table[0][j] + 1) / (self.words_in_a_class[0] + self.vocab) )
                    positive_pred += np.log( (self.table[1][j] + 1) / (self.words_in_a_class[1] + self.vocab))
            if positive_pred > negative_pred:
                predictions.append(1)
            else:
                predictions.append(0)
            

        return predictions

            
    def top10Words(self,feature_extractor):
        self.ratios = np.zeros(self.table[0].size)
        for i in range(self.ratios.size):
            self.ratios[i] = ((self.table[1][i]/self.n_documents_class[1])/(self.table[0][i]/self.n_documents_class[0]))

        lowest = self.ratios.argsort()[:10]
        highest = self.ratios.argsort()[-10:]

        print("Highest ratio words -------------")
        feature_extractor.indexesToWords(highest)
        print("Lowest ratio words -------------")
        feature_extractor.indexesToWords(lowest) 

        
        
            

# TODO: Implement this
class LogisticRegressionClassifier(HateSpeechClassifier):
    """Logistic Regression Classifier
    """
    def __init__(self):
        # Add your code here!
        self.iterations = 100
        self.learningRate = 0.1

    def fit(self, X, Y):
        # Add your code here!
        self.weights = np.zeros((X[0].size))
        self.bias = 0
        for epoch in range(self.iterations):
            for document in range(Y.size):
                L = ( 1 / (1 + np.exp( - X[document].dot(self.weights) + self.bias))) - Y[document]
                #self.bias = self.bias - self.learningRate * L
                L = np.dot(L,X[document])
                L = np.add(L,+0.0001*2*self.weights)
                self.weights = self.weights - ( self.learningRate * L )
                
            

    
    def predict(self, X):

        predictions = []

        for document in range(X.shape[0]):
            z = self.bias
            for feature in range(self.weights.size):
                z += self.weights[feature] * X[document][feature]
            
                probability = 1 / ( 1 + np.exp(- z ) )

            if probability > 0.5:
                predictions.append(1)
            else:
                predictions.append(0)

        return predictions

        


class PerceptronClassifier(HateSpeechClassifier):
    """Logistic Regression Classifier
    """

    def __init__(self):
        # Add your code here!
        raise Exception("Must be implemented")

    def fit(self, X, Y):
        # Add your code here!
        raise Exception("Must be implemented")

    def predict(self, X):
        # Add your code here!
        raise Exception("Must be implemented")

# you can change the following line to whichever classifier you want to use for bonus
# i.e to choose NaiveBayes classifier, you can write
# class BonusClassifier(NaiveBayes):
class BonusClassifier(LogisticRegressionClassifier):
    def __init__(self):
        super().__init__()
