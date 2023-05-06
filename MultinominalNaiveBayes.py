import math
import fractions

class MultinomialNaiveBayes:
    def __init__(self):
        self.classes = None
        self.vocab = None
        self.class_probabilities = None
        self.feature_probabilities = None
        self.word_count = None
        self.alpha = 1.0
    
    def train(self, X, y, alpha=1.0):
        
        self.alpha = alpha
        
        # Parse and store vocabulary
        self.vocab = list(set([item for sublist in X for item in sublist]))
                
        # Parse and store classes
        self.classes = list(set([item for sublist in y for item in sublist]))
        
        # Init probabilities
        self.class_probabilities = [0.0] * len(self.classes)
        self.feature_probabilities = [[0.0] * len(self.vocab)] * len(self.classes)
        self.word_count = [0] * len(self.classes)

        
        # Compute class probabilities & feature probabilities
        for class_index, curr_class in enumerate(self.classes):
            print(f"\r[MultinomialNB] Training class: {curr_class} {class_index+1}/{len(self.classes)}...\r", end="\r")
            class_occurences = sum(1 for i in range(len(y)) if curr_class in y[i])
            self.class_probabilities[class_index] =  class_occurences / len(y)
            
            print(f"Class {curr_class} has {class_occurences} occurences with probability {self.class_probabilities[class_index]}")

            # Get all documents of the current class
            x_in_class = [X[i] for i in range(len(y)) if curr_class in y[i]]
            class_concatenated = [item for sublist in x_in_class for item in sublist]
            self.word_count[class_index] = len(class_concatenated)


            # # Init feature probabilities for class

            for word_index, word in enumerate(self.vocab):
                word_occurrence_in_class = class_concatenated.count(word)
                self.feature_probabilities[class_index][word_index] = (word_occurrence_in_class + alpha) / (len(class_concatenated) + (alpha*len(self.vocab)))

            
    def predict(self, X):
        y_pred = [0] * len(X)
        for i, x in enumerate(X):
            print(f"\r[MultinomialNB] Predicting article {i}/{len(X)}\r", end="\r" )
            class_scores = [0.0] * len(self.classes)
            for class_index, curr_class in enumerate(self.classes):
                # Compute log-likelihood for each feature
                class_scores[class_index] = math.log(self.class_probabilities[class_index])      
                            
                
                for word in x:
                    if word in self.vocab:
                        word_index = self.vocab.index(word)
                        class_scores[class_index] += math.log(self.feature_probabilities[class_index][word_index])
                    else:
                        class_scores[class_index] += math.log(self.alpha /(self.alpha*len(self.vocab) + self.word_count[class_index]))
                
            # Choose class with highest log-probability
            # print(f"Classes: {self.classes}")
            # print(class_scores)
            y_pred[i] = self.classes[class_scores.index(max(class_scores))]
        print()
        return y_pred
