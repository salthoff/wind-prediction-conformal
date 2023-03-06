import numpy as np

def scoring(predictions, label, confidence):
    print(predictions)
    accuracy = np.sum((np.squeeze(label) > predictions[:,0]) & (np.squeeze(label) < predictions[:,1]))/len(label)
    width = np.mean(predictions[:,1]-predictions[:,0])
    return 1000 * np.abs(confidence-accuracy) + width


def model_runs(model_class, model_params, input, forecast, measurement, num_splits, confidence):
    score = np.empty(len(model_params))

    for i in range(len(model_params)):
        #predictions = np.array()
        first = True
        labels = np.array([])
        for s in range(num_splits):
            j = 0
            model = model_class(**model_params[i])
            j = int(1/(1-confidence)) + 1
            model.calibrate(input[:j],forecast[:j],measurement[:j])
            while j < len(measurement):
                if s > 0:
                    model.calibrate(input[j:np.min([j+s,len(measurement)])], forecast[j:np.min([j+s,len(measurement)])], measurement[j:np.min([j+s,len(measurement)])])
                if j+s < len(measurement):
                    pred = model.predict(input[j+s], forecast[j+s], confidence = confidence)
                    if first:
                        predictions = np.array([pred])
                        first = False
                    else:
                        predictions = np.r_[predictions, np.array([pred])]
                    labels = np.r_[labels, measurement[j+s]]
                    
                    if j + num_splits < len(measurement):
                        model.calibrate(input[j+s+1:j+num_splits], forecast[j+s+1:j+num_splits], measurement[j+s+1:j+num_splits])
                
                j += num_splits
        score[i] = scoring(predictions, labels, confidence)
    
    best_model = model_params[np.argmin(score)]
    return best_model


def train_schedule(model_class, model_parameters, baseinput, basefc, basems, testinput, testfc, testms, num_splits = 5, confidence = 0.9):
    predictions = np.empty((len(testms),2))
    models = []
    traininput = baseinput
    trainfc = basefc
    trainms = basems
    for i in range(len(testms)):
        best_model = model_runs(model_class, model_parameters, traininput, trainfc, trainms, num_splits, confidence)
        model = model_class(**best_model)
        model.calibrate(traininput,trainfc,trainms)
        models.append(best_model)
        predictions[i] = model.predict(testinput[i], testfc[i], confidence)
        traininput = np.r_[baseinput, testinput[:i+1]]
        trainfc = np.r_[basefc, testfc[:i+1]]
        trainms = np.r_[basems, testms[:i+1]]
    return predictions, models
        

    
    