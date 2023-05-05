import math
import fractions

class MultinomialNaiveBayes:
    def __init__(self):
        self.classes = None
        self.vocab = None
        self.class_probabilities = {}
        self.feature_probabilities = {}
    
    def train(self, X, y, alpha=1.0):
        
        # Parse and store vocabulary
        self.vocab = list(set([item for sublist in X for item in sublist]))
                
        # Parse and store classes
        self.classes = list(set([item for sublist in y for item in sublist]))
        
        # Compute class probabilities & feature probabilities
        for curr_class in self.classes:
            class_occurences = sum(1 for i in range(len(y)) if curr_class in y[i])
            self.class_probabilities[curr_class] =  class_occurences / len(y)

            # Get all documents of the current class
            x_in_class = [X[i] for i in range(len(y)) if curr_class in y[i]]
            class_concatenated = [item for sublist in x_in_class for item in sublist]

            # Init feature probabilities for class
            self.feature_probabilities[curr_class] = {}

            for word in self.vocab:
                word_occurrence_in_class = class_concatenated.count(word)
                self.feature_probabilities[curr_class][word] = (word_occurrence_in_class + alpha) / (len(class_concatenated) + (alpha*len(self.vocab)))

            
    def predict(self, X):
        y_pred = [0] * X.shape[0]
        for i, x in enumerate(X):
            class_scores = {}
            for curr_class in self.classes:
                # Compute log-likelihood for each feature
                class_scores[curr_class] = math.log(self.class_probabilities[curr_class])      
                
                for word in x:
                    if word in self.vocab:
                        class_scores[curr_class] += math.log(self.feature_probabilities[curr_class][word])
                
            # Choose class with highest log-probability
            y_pred[i] = max(class_scores, key=class_scores.get)
        return y_pred
