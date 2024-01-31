"""
Functions for fitting and testing classifier models
with k-folds CV.
"""


import numpy as np


def fit_svm(graphs_l, 
            labels,
            smfm,
            C: float = 2000.0,
            kernel: str = 'rbf', 
            degree: int = 3,
            n_splits: int = 10,
            random_state: int = 948724):
    """
    Gao et. al got 71.2 ± 3.25 accuracy
    """
    from sklearn import svm
    from sklearn.model_selection import StratifiedKFold

    # k-folds CV
    skf = StratifiedKFold(n_splits=n_splits,
                          shuffle=True,
                          random_state=random_state)
    splits = skf.split(X=graphs_l, y=labels)
    
    model = svm.SVC(C=C, kernel=kernel, degree=degree)
    fold_accs = [None] * n_splits
    
    # for each fold:
    for i, (train_index, test_index) in enumerate(splits):
        # get train and test Xs and ys
        train_X = smfm[np.array(train_index)]
        test_X = smfm[np.array(test_index)]
        train_y = labels[train_index]
        test_y = labels[test_index]
    
        # fit model; get predictions on test set
        model.fit(train_X, train_y)
        preds = model.predict(test_X)
    
        # calc test accuracy and save
        acc = np.sum(preds == test_y) / len(test_y)
        fold_accs[i] = acc

    return fold_accs


def print_cv_res(fold_accs):
    """
    Calcs and prints cv-mean accuracy and st. dev.
    """
    import matplotlib.pyplot as plt

    cv_mean_acc = np.mean(fold_accs)
    cv_stdev_acc = np.std(fold_accs)
    
    plt.hist(fold_accs, color='gray')
    plt.locator_params(axis="y", integer=True, tight=True)
    # mean line
    plt.axvline(x=cv_mean_acc, color='red')
    # st dev bar
    l_stdev_x = cv_mean_acc - cv_stdev_acc
    r_stdev_x = cv_mean_acc + cv_stdev_acc
    plt.plot([l_stdev_x, r_stdev_x], [1.5, 1.5], color='red')
    plt.plot([l_stdev_x, l_stdev_x], [1.45, 1.55], color='red')
    plt.plot([r_stdev_x, r_stdev_x], [1.45, 1.55], color='red')
    plt.show()

    print(f'{len(fold_accs)}-fold CV accuracy (%):')
    print('------------------------')
    print(f'mean     {cv_mean_acc*100:.1f}')
    print(f'st dev   {cv_stdev_acc*100:.1f}')
    print(f'range    {np.min(fold_accs)*100:.1f} - {np.max(fold_accs)*100:.1f}')

    print(f"\n{'fold(k)':<9}{'accuracy(%)'}")
    print('--------------------')
    for i, a in enumerate(fold_accs):
        print(f"{i+1:<9}{a*100:.1f}")

