def negation_evaluation(x_test,y_test,classifier):
    '''
    negation_evaluation shows the performances of the classifier when it has to classify utterances with negations
    input:      list of utterances to classify, set of true labels, classifier
    output:     f1 macro, f1 micro, recall macro, recall micro, precision macro, precision micro, confusion matrix
    
    '''
    real_classes = []
    utterances_test = []
    #select utterances which contains negations avoiding "don't care" utterances
    for x,y in zip(x_test.to_numpy(),y_test.to_numpy()):
        if ("dont" in x or "do not" in x ) and "care" not in x:
            real_classes.append(y)
            utterances_test.append(x)
            
    #preprocess utterances and predict class
    classes_pred_test = []
    for xx in utterances_test:
        x_trans = vectorizer.transform([preprocess_data(xx)]).toarray()
        classes_pred_test.append(classifier.predict(x_trans))
    
    return evaluation(classes_pred_test, real_classes)
