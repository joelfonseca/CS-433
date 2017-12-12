import numpy as np
from PIL import Image
from tqdm import tqdm
import glob

from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn import linear_model

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from loader import TrainingSet, TestSet
from model import CompleteCNN
from utils import prediction_to_np_patched2, patched_to_submission_lines, concatenate_images, train_valid_split
from postprocessing import majority_voting2
from parameters import K_FOLD, SEED

import pprint

PREDICTION_TEST_DIR = 'predictions_test/'
SAVED_MODEL_DIR = 'saved_models/'

def compute_input(data, models):
    X = []
    for i, model in enumerate(models):
        if i == 0:
            X = np.c_[model.predict(data).data.view(-1).cpu().numpy()]
        else:
            X = np.c_[X, model.predict(data).data.view(-1).cpu().numpy()]

    return X

TRAIN = False

if __name__ == '__main__':

    # load all the best models from grid search
    all_models = glob.glob('saved_models/*_best.pt')
    print('{} models loaded from grid search.' .format(len(all_models)))

    """
    regr30 = joblib.load('regr30.pkl')
    c = regr30.coef_
    #print(c)

    model_weight = list(zip(all_models, c))
    model_weight.sort(key=lambda x: -x[1])
    #pprint.pprint(model_weight)

    top10_models = [x[0] for x in model_weight[:10]]
    #pprint.pprint(top10_models)
    """

    models = []
    for model_name in all_models:
        tmp = model_name.split('_')
        model = CompleteCNN(float(tmp[5]), tmp[6])
        model.load_state_dict(torch.load(model_name))
        model.cuda()
        model.eval()
        models.append(model)

    if TRAIN:

        train_loader = DataLoader(TrainingSet(), num_workers=4, batch_size=1, shuffle=False)

        # Create training and validation split
        train_data, train_targets, valid_data, valid_targets = train_valid_split(train_loader, K_FOLD, SEED)

        # Combine train/validation data and targets as tuples
        train_data_and_targets = list(zip(train_data, train_targets))
        valid_data_and_targets = list(zip(valid_data, valid_targets))

        X_train = []
        y_train = []
        for i, (data, target) in enumerate(tqdm(train_data_and_targets)):
            if i == 0:
                X_train = np.r_[compute_input(data, models)]
                y_train = np.r_[target.data.view(-1).cpu().numpy()]
            else:
                X_train = np.r_[X_train, compute_input(data, models)]
                y_train = np.r_[y_train, target.data.view(-1).cpu().numpy()]
        
        #np.save('X_train.npy', X_train)
        #np.save('y_train.npy', y_train)

        X_valid = []
        y_valid = []
        for i, (data, target) in enumerate(tqdm(valid_data_and_targets)):
            if i == 0:
                X_valid = np.r_[compute_input(data, models)]
                y_valid = np.r_[target.data.view(-1).cpu().numpy()]
            else:
                X_valid = np.r_[X_valid, compute_input(data, models)]
                y_valid = np.r_[y_valid, target.data.view(-1).cpu().numpy()]

        #np.save('X_valid.npy', X_valid)
        #np.save('y_valid.npy', y_valid)

        regr = linear_model.LinearRegression()  
        regr.fit(X_train, y_train)

        # save the classifier
        joblib.dump(regr, 'regr.pkl')   

        acc = accuracy_score(y_valid, regr.predict(X_valid).round())
        print('Accuracy: {:.5f}.' .format(acc))

    else:

        regr = joblib.load('regr.pkl')
        c = regr.coef_
        print(c)


        test_loader = DataLoader(TestSet(), num_workers=4, batch_size=1, shuffle=False)
        lines = []

        y_preds = []
        flips = [] 
        X_test = []
        for i, (data, _) in enumerate(tqdm(test_loader)):
            
            X_test = compute_input(Variable(data, volatile=True).cuda(), models)
            y_pred = regr.predict(X_test).reshape((608, 608))

            y_preds.append(y_pred)
            flips.append(data)

            if (i+1)%4 == 0:
                
                kaggle_pred = prediction_to_np_patched2(majority_voting2(y_preds))
                
                # Save the prediction image (concatenated with the real image)
                concat_data = concatenate_images(flips[i-3].squeeze().permute(1, 2, 0).numpy(), kaggle_pred * 255)
                Image.fromarray(concat_data).convert('RGB').save(PREDICTION_TEST_DIR + 'prediction_' + str((i+1)//4) + '.png')

                # Store the lines in the form Kaggle wants it: "{:03d}_{}_{},{}"
                for new_line in patched_to_submission_lines(kaggle_pred, ((i+1)//4)):
                    lines.append(new_line)
                
                y_preds = []
                
        # Create submission file
        with open('data/submissions/submission_stacking_top10_mv.csv', 'w') as f:
            f.write('id,prediction\n')
            f.writelines('{}\n'.format(line) for line in lines)

        print('Predictions done.')