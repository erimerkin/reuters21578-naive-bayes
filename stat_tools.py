def confusion_matrix(y_true, y_predict):
    
    class_labels = list(set([item for sublist in y_true for item in sublist]))
    
    conf_matrix = {label: {"true_positive": 0, "false_positive": 0, "false_negative": 0, "true_negative": 0} for label in class_labels}
    
    
    for index, true_labels in enumerate(y_true):
        for select_class in class_labels:
            
            if select_class in true_labels:
                if select_class in y_predict[index]:
                    conf_matrix[select_class]["true_positive"] += 1
                else:
                    conf_matrix[select_class]["false_negative"] += 1
            else:
                if select_class in y_predict[index]:
                    conf_matrix[select_class]["false_positive"] += 1
                else: 
                    conf_matrix[select_class]["true_negative"] += 1
                    
    return conf_matrix
    

def f1_score(y_true, y_predict, type="macro"):
    
    conf_matrix = confusion_matrix(y_true, y_predict)
    
    calculated_precision = 0
    calculated_recall = 0 
    
    
    if type == "macro":
        class_recall = [0.0] * len(conf_matrix)
        class_precision = [0.0] * len(conf_matrix)

        for index, class_label in enumerate(conf_matrix):
            class_recall[index] = conf_matrix[class_label]["true_positive"] / (conf_matrix[class_label]["true_positive"] + conf_matrix[class_label]["false_negative"])
            class_precision[index] = conf_matrix[class_label]["true_positive"] / (conf_matrix[class_label]["true_positive"] + conf_matrix[class_label]["false_positive"])

        calculated_precision = sum(class_precision) / len(class_precision)
        calculated_recall = sum(class_recall) / len(class_recall)
        
    elif type == "micro":
        total_true_positive = sum([conf_matrix[class_label]["true_positive"] for class_label in conf_matrix])
        total_false_positive = sum([conf_matrix[class_label]["false_positive"] for class_label in conf_matrix])
        total_false_negative = sum([conf_matrix[class_label]["false_negative"] for class_label in conf_matrix])
        
        calculated_precision = total_true_positive / (total_true_positive + total_false_positive)
        calculated_recall = total_true_positive / (total_true_positive + total_false_negative)
        
    else:
        Exception("Invalid type for F1 score")
        
    return 2 * (calculated_precision * calculated_recall) / (calculated_precision + calculated_recall)
    