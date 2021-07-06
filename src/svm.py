import numpy as np
import os
import random
from sklearn.metrics import hinge_loss
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC, LinearSVC
import torch
import time


def compute_metrics(y, pred, num_classes=10):
    """
    Compute perfomance metrics given the predicted labels and the true labels
    Args:
        y: True label vector
           (Type: np.ndarray)
        pred: Predicted label vector
              (Type: np.ndarray)
    Returns:
        metrics: Metrics dictionary
                 (Type: dict[str, *])
    """
    # Make sure everything is a numpy array
    if isinstance(y, torch.Tensor):
        y = y.cpu().data.numpy()
    elif not isinstance(y, np.ndarray):
        y = np.array(y)
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().data.numpy()
    elif not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    assert isinstance(y, np.ndarray)
    assert isinstance(pred, np.ndarray)

    # Convert from one-hot to integer encoding if necessary
    if y.ndim == 2:
        y = np.argmax(y, axis=1)
    if pred.ndim == 2:
        pred = np.argmax(pred, axis=1)
    assert y.ndim == 1
    assert pred.ndim == 1

    acc = (y == pred).mean()

    class_acc = []
    for class_idx in range(num_classes):
        idxs = (y == class_idx)
        class_acc.append((y[idxs] == pred[idxs]).mean())

    ave_class_acc = np.mean(class_acc)

    return {
        'accuracy': acc,
        'class_accuracy': class_acc,
        'average_class_accuracy': ave_class_acc
    }


def aggregrate_video_accuracy(softmaxes, labels):
    correct = 0
    total_vids = 0
    for vid_idx in softmaxes.keys():
        mean_softmax_np = np.mean(np.stack(softmaxes[vid_idx]), axis=0)
        correct += int(mean_softmax_np.argmax() == labels[vid_idx])
        total_vids += 1
    vid_accuracy = float(correct) / float(total_vids)
    return vid_accuracy


def train_svm(train_data, valid_data, test_data, standardize=False, model_dir='.', C=1.0, kernel='rbf',
              num_classes=10, tol=0.001, max_iterations=-1, verbose=False,
              random_state=12345678, val_indices=8, train_classes=None, **kwargs):
    """
    Train a Support Vector Machine model on the given data
    Args:
        X_train: Training feature data
                 (Type: np.ndarray)
        y_train: Training label data
                 (Type: np.ndarray)
        X_test: Testing feature data
                (Type: np.ndarray)
        y_test: Testing label data
                (Type: np.ndarray)
    Keyword Args:
        C: SVM regularization hyperparameter
           (Type: float)
        verbose:  If True, print verbose messages
                  (Type: bool)
    Returns:
        clf: Classifier object
             (Type: sklearn.svm.SVC)
        y_train_pred: Predicted train output of classifier
                     (Type: np.ndarray)
        y_test_pred: Predicted test output of classifier
                     (Type: np.ndarray)
    """
    np.random.seed(random_state)
    random.seed(random_state)

    # Standardize features
    if standardize:
        print('Standardizing features...')
        stdizer = StandardScaler()
        train_data['features'] = stdizer.fit_transform(train_data['features'])
        if valid_data:
            valid_data['features'] = stdizer.transform(valid_data['features'])
        if test_data:
            test_data['features'] = stdizer.transform(test_data['features'])

    # Encdoe labels
    le = LabelEncoder()
    le.fit(train_data['labels'])
    train_data['labels'] = le.transform(train_data['labels'])
    if valid_data:
        valid_data['labels'] = le.transform(valid_data['labels'])
    if test_data:
        test_data['labels'] = le.transform(test_data['labels'])

    # Get train data
    X_train = train_data['features']
    y_train = train_data['labels']

    # Create classifier
    '''
    clf = LinearSVC(
        penalty='l2', # the norm used in the penalization, l2 is the standard
        loss='squared_hinge', # ‘hinge’ is the standard SVM loss
        dual=False, # True, #  prefer dual=False when n_samples > n_features.
        tol=1e-4, # Tolerance for stopping criteria, def=1e-4
        C=C, # Regularization parameter. regularization is inversely proportional to C.
        multi_class='ovr', # trains n_classes one-vs-rest classifiers
        verbose=False,
        max_iter=max_iterations # The maximum number of iterations to be run.
    )
    '''
    clf = LinearSVC(C=C)

    # Fit data and get output for train and valid batches
    print('Fitting model to data...')
    start = time.time()
    clf.fit(X_train, y_train)
    print(f"Time to fit model: {time.time() - start}")

    # Compute new metrics
    print("Perfoming predictions on Train Set", flush=True)
    start = time.time()
    y_train_pred = clf.predict(X_train)
    print(f"Time to do predictions on Train Set: {time.time() - start}", flush=True)
    if train_classes is not None:
        classes = train_classes
    else:
        classes = np.arange(num_classes)
    train_loss = hinge_loss(y_train, clf.decision_function(X_train), labels=classes)
    train_metrics = compute_metrics(y_train, y_train_pred, num_classes=num_classes)
    train_metrics['loss'] = train_loss
    train_msg = 'Train - hinge loss: {}, acc: {}'
    print(train_msg.format(train_loss, train_metrics['accuracy']), flush=True)
    
    if valid_data:
        print("Perfoming predictions on Val Set", flush=True)
        start = time.time()
        X_valid = valid_data['features']
        y_valid = valid_data['labels']
        y_valid_pred = clf.predict(X_valid)
        print(f"Time to do predictions on Val Set: {time.time() - start}")
        valid_loss = hinge_loss(y_valid, clf.decision_function(X_valid), labels=classes)
        valid_metrics = compute_metrics(y_valid, y_valid_pred, num_classes=num_classes)
        valid_metrics['loss'] = valid_loss
        valid_msg = 'Valid - hinge loss: {}, acc: {}'
        print(valid_msg.format(valid_loss, valid_metrics['accuracy']), flush=True)
    else:
        valid_metrics = {}

    # Evaluate model on test data
    if test_data:
        test_metrics = {}
        print("Perfoming predictions on Test Set", flush=True)
        start = time.time()
        X_test = test_data['features']
        y_test = test_data['labels']
        y_test_pred_frame = clf.decision_function(X_test) #predict_proba(X_test)
        print(f"Time to do predictions on Test Set: {time.time() - start}", flush=True)
        test_loss = hinge_loss(y_test, clf.decision_function(X_test), labels=classes)
        y_test_pred = []
        y_test_gt = []
        softmaxes = {}
        labels = {}
        # Video Level accuracy
        for j in range(len(val_indices)):
            video_id  = val_indices[j]
            sm = y_test_pred_frame[j]
            label = y_test[j]
            # append it to video dict
            softmaxes.setdefault(video_id, []).append(sm)
            labels[video_id] = label
        vid_accuracy = aggregrate_video_accuracy(softmaxes, labels)
        test_metrics['accuracy'] = vid_accuracy
    else:
        test_metrics = {}
    return clf, train_metrics, valid_metrics, test_metrics
