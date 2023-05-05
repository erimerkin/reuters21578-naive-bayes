def calculate_recall(results, actual_results, average = 'macro'):
    if average=='macro':
        pass
    elif average=='micro':
        pass
    else:
        Exception('Invalid average type')
    
    

def calculate_precision(results, actual_results, average = 'macro'):
    if average=='macro':
        pass
    elif average=='micro':
        pass
    else:
        Exception('Invalid average type')
    

def calculate_f1_score(results, actual_results, average = 'macro'):
    
    if average=='macro':
        return calculate_f1_score_macro(results, actual_results)
    elif average=='micro':
        return calculate_f1_score_micro(results, actual_results)
    
    
    precision = calculate_precision(results, actual_results)
    recall = calculate_recall(results, actual_results)
    
    return 2 * (precision * recall) / (precision + recall)
    

def compare_results(results, actual_results):
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0