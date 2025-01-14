#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from skimage.io import imread
from skimage.transform import resize

import mlflow

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll.base import scope


# In[ ]:


mlflow.set_experiment("MRI-classification")


# In[3]:


def load_image_files(container_path, dimension=(64, 64)):
    """
    Load image files with categories as subfolder names 
    which performs like scikit-learn sample dataset
    
    Parameters
    ----------
    container_path : string or unicode
        Path to the main folder holding one subfolder per category
    dimension : tuple
        size to which image are adjusted to
        
    Returns
    -------
    Bunch
    """
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]

    descr = "A image classification dataset"
    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            img = imread(file)
            img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
            flat_data.append(img_resized.flatten()) 
            images.append(img_resized)
            target.append(i)
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)

    return Bunch(data=flat_data,
                 target=target,
                 target_names=categories,
                 images=images,
                 DESCR=descr)


# In[4]:


image_dataset = load_image_files("Brain-Tumor-Classification-DataSet/Training")


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(
    image_dataset.data, image_dataset.target, test_size=0.3,random_state=109)


# In[6]:


def objective(params):
    with mlflow.start_run():
        classifier_type = params['type']
        mlflow.set_tag("model", "sklearn_"+classifier_type)
        mlflow.log_params(params)
        del params['type']
        if classifier_type == 'svm':
            clf = SVC(**params)
        elif classifier_type == 'rf':
            clf = RandomForestClassifier(**params)
        else:
            return 0
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_pred, y_test)
        mlflow.log_metric('accuracy', accuracy)

        # Logging artifacts takes more space
        #mlflow.sklearn.log_model(clf, artifact_path="models")
    
    # Because fmin() tries to minimize the objective, this function must return the negative accuracy. 
    return {'loss': accuracy, 'status': STATUS_OK}


# In[7]:


search_space = hp.choice('classifier_type', [
    {
        'type': 'svm',
        'C': hp.lognormal('SVM_C', 0, 1.0),
        'kernel': hp.choice('kernel', ['linear', 'rbf'])
    },
    {
        'type': 'rf',
        'max_depth': scope.int(hp.quniform('max_depth', 2, 5, 1)),
        'criterion': hp.choice('criterion', ['gini', 'entropy'])
    },
])


# In[ ]:


mlflow.sklearn.autolog()

best_result = fmin(
    fn=objective, 
    space=search_space,
    algo=tpe.suggest,
    max_evals=32,
    trials=Trials()
    )

