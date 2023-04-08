import numpy as np
from tqdm import tqdm

def scoring(predictions, label, confidence):
    #print(predictions)
    accuracy = np.sum((np.squeeze(label) > predictions[:,0]) & (np.squeeze(label) < predictions[:,1]))/len(label)
    width = np.mean(predictions[:,1]-predictions[:,0])
    return 2000 * np.abs(confidence-accuracy) + width

def crps(predictions, label):
    scores = np.empty(len(label))
    length = predictions.shape[1]
    
    for i in range(len(label)):
        low = predictions[i,:][predictions[i,:]<=label[i]]
        up = predictions[i,:][predictions[i,:]>=label[i]]
        f_low = np.cumsum((1/length)*np.ones(len(low)))
        if f_low.size > 0:
            f_up = np.cumsum((1/length)*np.ones(len(up))) + f_low[-1]
        else:
            f_up = np.cumsum((1/length)*np.ones(len(up)))
        low_score = np.sum((np.power(f_low[:-1],2)+np.power(f_low[1:],2))*(low[1:]-low[:-1]))/2
        up_score = np.sum((np.power((f_up[:-1]-1),2)+np.power((f_up[1:]-1),2))*(up[1:]-up[:-1]))/2
        scores[i] = low_score + up_score

    return np.mean(scores)



def model_runs(model_class, model_params, input, forecast, measurement, num_splits, len_distr, block_training = False):
    score = np.empty(len(model_params))
    if block_training:
        block_len = len(forecast) // num_splits
        rest = len(forecast) % num_splits
        for i in range(len(model_params)):
            first = True
            labels = np.array([])
            for j in range(num_splits):
                block_input = np.r_[input[rest+(j+1)*block_len:] ,input[:j*block_len+rest]]
                block_fc = np.r_[forecast[rest+(j+1)*block_len:] ,forecast[:j*block_len+rest]]
                block_ms = np.r_[measurement[rest+(j+1)*block_len:] ,measurement[:j*block_len+rest]]
                model = model_class(**model_params[i])
                model.calibrate(block_input, block_fc, block_ms)
                test_input = input[j*block_len+rest:rest+(j+1)*block_len]
                test_fc = forecast[j*block_len+rest:rest+(j+1)*block_len]
                test_ms = measurement[j*block_len+rest:rest+(j+1)*block_len]
                for t in range(block_len):
                    pred = model.predict(test_input[t],test_fc[t],length_distr = len_distr)
                    if first:
                        predictions = np.array([pred])
                        first = False
                    else:
                        predictions = np.r_[predictions, np.array([pred])]
                    
                    labels = np.r_[labels, test_ms[t]]
            score[i]=crps(predictions, labels)
        best_model = model_params[np.argmin(score)]
        return best_model

    for i in range(len(model_params)):
        first = True
        labels = np.array([])
        
        model = model_class(**model_params[i])
        rest = len(forecast) % num_splits
        j = rest + 1
        model.calibrate(input[:j],forecast[:j],measurement[:j])
        while j < len(measurement):
            pred = model.predict(input[j], np.array(forecast[j]), length_distr = len_distr)
            model.calibrate(input[j:j+1],forecast[j:j+1],measurement[j:j+1])
            if first:
                predictions = np.array([pred])
                first = False
            else:
                predictions = np.r_[predictions, np.array([pred])]
            labels = np.r_[labels, measurement[j]]
            j += 1
        score[i]=crps(predictions, labels)
    
    best_model = model_params[np.argmin(score)]
    return best_model


def train_schedule(model_class, model_parameters, baseinput, basefc, basems, testinput, testfc, testms, num_splits = 5, len_distr = 200, block_training = False):
    predictions = np.empty((len(testms),len_distr))
    models = []
    traininput = baseinput
    trainfc = basefc
    trainms = basems
    for i in tqdm(range(len(testms))):
        best_model = model_runs(model_class, model_parameters, traininput, trainfc, trainms, num_splits, len_distr, block_training=block_training)
        model = model_class(**best_model)
        model.calibrate(traininput,trainfc,trainms)
        models.append(best_model)
        predictions[i] = model.predict(testinput[i], testfc[i], length_distr=len_distr)
        traininput = np.r_[baseinput, testinput[:i+1]]
        trainfc = np.r_[basefc, testfc[:i+1]]
        trainms = np.r_[basems, testms[:i+1]]
    return predictions, models
        

    
    