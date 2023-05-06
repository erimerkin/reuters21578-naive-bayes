import os
import re
import sys
import string

def tokenize(input, stopwords):
    
    punctuation_removed_content = input.translate(str.maketrans('', '', string.punctuation))        
    
    lowercased_content = punctuation_removed_content.lower()
    tokenized_content = lowercased_content.split()
    
    return [token for token in tokenized_content if token not in stopwords]


def load_stopwords(stopwords_path):
    try:
        if not os.path.isfile(stopwords_path):
            raise Exception("[ERROR] The given stopwords path is not a file!")
        
        stopwords = []
        
        with open(stopwords_path, "r") as stopwords_file:
            for line in stopwords_file:
                stopwords += line.split()
                
        stopwords.append("Reuter")
        return stopwords
    
    except Exception as e:
        quit(e)
    
def load_articles(dataset_path, stopwords_path):
    
    print("[INFO] Loading Reuters21578 Dataset...")
    
    try: 
        
        # Check if the given path is a directory
        if not os.path.isdir(dataset_path):
            raise Exception("[ERROR] The given dataset path is not a directory!")
        
        stopwords = load_stopwords(stopwords_path)
                    
        processed_files = 0
        
        training_set = []
        test_set = []
        
        for root, _, files in os.walk(os.path.join(dataset_path)):
            
            total_target_files = len([file for file in files if file.endswith(".sgm")])
            
            for file in files:
                if file.endswith(".sgm"):
                    processed_files += 1
                    
                    print(f"\r[INFO] Processing file {file} {processed_files}/{total_target_files}...", end="\r")
                    
                    # Read the file
                    current_file = open(os.path.join(root, file), "r", encoding="latin-1")
                    
                    # Split the file into articles
                    articles = re.split(r'</REUTERS>', current_file.read())

                    # Iterate over the articles
                    for article in articles:
                        
                        # Remove starting and ending whitespaces
                        target_article = article.strip()
                        
                        # If article is empty, skip it
                        if target_article == "":
                            continue
                                                
                        # Learn if article is training or test set
                        target_set = re.search(r'LEWISSPLIT="(.*?)"', target_article).group(1)
                                            
                        # GET TOPICS of article
                        topics = re.search(r'<TOPICS>(.+)</TOPICS>', target_article, re.DOTALL)
                        
                        topic_list = []

                        if topics:
                                                    
                            topic_list = topics.group(1).split("<D>")
                            topic_list = [topic.replace("</D>", "") for topic in topic_list if topic != ""]
                    
                            # Parse the content of the article
                            corpus = ""
                            number_removed_text = re.search(r'<TEXT(.+)</TEXT>', target_article, re.DOTALL).group(1)
                            
                            # number_removed_text = re.sub(r'\d+', '', text_part)
                            
                            if title := re.search(r'<TITLE>(.+)</TITLE>', number_removed_text, re.DOTALL):
                                corpus += title.group(1)
                            
                            if body := re.search(r'<BODY>(.+)</BODY>', number_removed_text, re.DOTALL): 
                                corpus += " " + body.group(1)

                            # If corpus is empty, skip it
                            if corpus == "":
                                continue

                            # Tokenize the content of the article
                            tokenized_content = tokenize(corpus, stopwords)
                            
                                                    
                            if target_set == "TEST":
                                test_set.append((tokenized_content, topic_list))
                            elif target_set == "TRAIN" :
                                training_set.append((tokenized_content, topic_list))                        

        print(f"\n[INFO] Training set size: {len(training_set)}")
        print(f"[INFO] Test set size: {len(test_set)}")
        
        return training_set, test_set
    except Exception as e:
        quit(e)
        
from MultivariateBernoulliNaiveBayes import MultivariateBernoulliNaiveBayes
from MultinominalNaiveBayes import MultinomialNaiveBayes

def collect_occurences(target_labels):
    occurences = {}
    
    for topics in target_labels:
        for label in topics:
            if label in occurences:
                occurences[label] += 1
            else:
                occurences[label] = 1
            
    return occurences


import random
import math
from stat_tools import f1_score, confusion_matrix

def main():

    
    training_set, test_set = load_articles(dataset_path=sys.argv[1], stopwords_path=sys.argv[2])
    
    
    x_train = [article[0] for article in training_set]
    y_train = [article[1] for article in training_set]
    
    x_test = [article[0] for article in test_set]
    y_test = [article[1] for article in test_set]
    
    #find most popular topics
    train_label_occurences = collect_occurences(y_train)
    test_label_occurences = collect_occurences(y_test)
    
    total_label_occurences = {key: train_label_occurences.get(key, 0) + test_label_occurences.get(key, 0) for key in set(train_label_occurences) | set(test_label_occurences)}
                
    sorted_topic_occurences = sorted(total_label_occurences.items(), key=lambda x: x[1], reverse=True)
    
    print(sorted_topic_occurences[:10])
    
    print("Most popular topics in training set:\nTOPIC\t\t|\tTRAIN\t|\tTEST")
    print("--" * 25)
    
    selected_x_train = []
    selected_y_train = []
    
    selected_x_test = []
    selected_y_test = []
    
    selected_topics = []
    for topic in sorted_topic_occurences[:10]:
        topic_key = topic[0]
        selected_topics.append(topic_key)
        print(f"{topic_key}\t\t|\t {train_label_occurences[topic_key]} \t|\t {test_label_occurences[topic_key]}")
        
    for i in range(len(x_train)):
        included_topics = [topic for topic in selected_topics if topic in y_train[i]]
        if any(included_topics):
            selected_x_train.append(x_train[i])
            selected_y_train.append(included_topics)
            
    for i in range(len(x_test)):
        included_topics = [topic for topic in selected_topics if topic in y_test[i] ]
        if any(included_topics):
            selected_x_test.append(x_test[i])
            selected_y_test.append(included_topics)

        
    # Divide train set into %20 development set and %80 training set
    random.seed(8)
    development_set_indexes = random.sample(range(len(selected_x_train)), int(len(selected_x_train) * 0.2))
        
    x_development = [selected_x_train[i] for i in development_set_indexes]
    y_development = [selected_y_train[i] for i in development_set_indexes]
    
    print(f"\n[INFO] Development set size: {len(x_development)}")
    
    tuning_x_train = [selected_x_train[i] for i in range(len(selected_x_train)) if i not in development_set_indexes]
    tuning_y_train = [selected_y_train[i] for i in range(len(selected_y_train)) if i not in development_set_indexes]
        
    multinominal_nb = MultivariateBernoulliNaiveBayes()
    
    alpha = 2.0
    
    multinominal_nb.train(tuning_x_train, tuning_y_train, alpha=alpha)
    print("Training done!")
    results = multinominal_nb.predict(x_development)
    
    
    macro_F1_score = f1_score(y_development, results, type="macro")
    micro_F1_score = f1_score(y_development, results, type="micro")
    print(f"F1 Score Macro: {macro_F1_score}, Micro: {micro_F1_score}, alpha: {alpha}")
    # multivariate_nb = MultivariateBernoulliNaiveBayes()
    
    
    # conf_matrix = confusion_matrix(y_development[25:50], ['earn'] * 25)
    
    # print(y_development[25:50])
    # print('-' * 25)    
    # print(conf_matrix)
    
    
    


if __name__ == "__main__":
    
    if len(sys.argv) != 3:
        quit("Usage: python index_builder.py <dataset_path> <stopwords_path>")
        
    main()

                    