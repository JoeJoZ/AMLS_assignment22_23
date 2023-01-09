import os
import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


def read_data(data_dir):
    X = []
    Y = []
    
    # load labels
    df = pd.read_csv(os.path.join(data_dir, 'labels.csv'), sep='\t')
    img_list = df['file_name'].values.tolist()
    label_list = df['eye_color'].values.tolist()
    
    # load images
    for i in range(len(img_list)):
        image = cv2.imread(os.path.join(data_dir, 'img', img_list[i]))
        image = image[220:300, 160:240]
        image = cv2.resize(image, (20, 20), interpolation=cv2.INTER_LINEAR).reshape(-1)
        result = image / 255.0
        X.append(result)
        Y.append(label_list[i])

    return X, Y


if __name__ == "__main__":
    # load data
    X_train, Y_train = read_data('Datasets/cartoon_set')
    X_test, Y_test = read_data('Datasets/cartoon_set_test')

    # model
    model = RandomForestClassifier(n_estimators=100)
    # trian
    model.fit(X_train, Y_train)
    
    # test
    Y_predict_train = model.predict(X_train)
    # Calculate the performance
    acc_train = accuracy_score(Y_train, Y_predict_train)

    Y_predict_test = model.predict(X_test)
    acc_test = accuracy_score(Y_test, Y_predict_test)

    cm = confusion_matrix(Y_test, Y_predict_test)
    print(cm)
    print('Acc_train: ', acc_train)
    print('Acc_test: ', acc_test)
