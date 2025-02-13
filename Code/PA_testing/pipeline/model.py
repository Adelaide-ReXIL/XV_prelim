
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC



def XGBoost(X_tr, Y_tr, X_te, Y_te,quick=False):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
    }

    # Create the model
    xgb_model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        tree_method='hist',
        random_state=42
    )

    if quick:
            best_model = xgb_model
            best_model.fit(X_tr, Y_tr)

            xgb_pred = best_model.predict(X_te)


            ConfusionMatrixDisplay.from_estimator(estimator=best_model, X=X_te, y=Y_te)

            print("Classification Report: XGBoost")
            report = classification_report(Y_te, xgb_pred, digits=2)
            print(report)
            return best_model


    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='accuracy',  
        cv=10,  
        verbose=1,
        n_jobs=-1  
    )

  
    grid_search.fit(X_tr, Y_tr)

   
    best_model = grid_search.best_estimator_
    best_model.fit(X_tr, Y_tr)

    xgb_pred = best_model.predict(X_te)

    ConfusionMatrixDisplay.from_estimator(estimator=best_model, X=X_te, y=Y_te)

    print("Classification Report: XGBoost")
    report = classification_report(Y_te, xgb_pred, digits=2)
    print(report)
    return best_model
def XGBoostP(X_tr, Y_tr, X_te, Y_te,quick=False):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
    }

    # Create the model
    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        tree_method='hist',
        random_state=42
    )

    if quick:
            best_model = xgb_model
            best_model.fit(X_tr, Y_tr)

            xgb_pred = best_model.predict(X_te)
            xgb_prob = best_model.predict_proba(X_te)
            for y,pred,prob in zip(Y_te,xgb_pred,xgb_prob):
                  print(f"True: {y}, Pred: {pred}, Probabilities: {prob}")

            xgb_prob=[max(a) for a in xgb_prob]

           
            


            return best_model


    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='accuracy',  
        cv=10,  
        verbose=1,
        n_jobs=-1  
    )

  
    grid_search.fit(X_tr, Y_tr)

   
    best_model = grid_search.best_estimator_
    best_model.fit(X_tr, Y_tr)

    xgb_pred = best_model.predict(X_te)
    

    ConfusionMatrixDisplay.from_estimator(estimator=best_model, X=X_te, y=Y_te)

    print("Classification Report: XGBoost")
    report = classification_report(Y_te, xgb_pred, digits=2)
    print(report)
    return best_model
def RF(X_tr, Y_tr, X_te, Y_te):

    param_grid = {
        'n_estimators': [int(x) for x in np.linspace(start=10, stop=100, num=10)],  
        'max_depth': range(1, 20),  
        'criterion': ['gini', 'entropy'] 
    }


    optimal_params = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=10, 
        scoring='accuracy',
        verbose=0,
        n_jobs=-1
    )


    optimal_params.fit(X_tr, Y_tr)
    print("Best parameters found: ", optimal_params.best_params_)


    criterion = optimal_params.best_params_['criterion']
    max_depth = optimal_params.best_params_['max_depth']
    n_estimators = optimal_params.best_params_['n_estimators']


    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        criterion=criterion,
        random_state=42
    )

    rf_model.fit(X_tr, Y_tr)


    rf_pred = rf_model.predict(X_te)


    ConfusionMatrixDisplay.from_estimator(estimator=rf_model, X=X_te, y=Y_te)

    print("Best Cross-Validation Score:",optimal_params.best_score_)
    print("Classification Report: Random Forest")
    print(classification_report(Y_te, rf_pred, digits=2))
    return rf_model


def NaiveBayes(X_tr, Y_tr, X_te, Y_te, quick=False):
    param_grid = {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]  
    }


    nb_model = GaussianNB()

    if quick:

        best_model = nb_model
        best_model.fit(X_tr, Y_tr)


        nb_pred = best_model.predict(X_te)
        ConfusionMatrixDisplay.from_estimator(estimator=best_model, X=X_te, y=Y_te)

        print("Classification Report: Naive Bayes (GaussianNB)")
        report = classification_report(Y_te, nb_pred, digits=2)
        print(report)
        return best_model


    grid_search = GridSearchCV(
        estimator=nb_model,
        param_grid=param_grid,
        scoring='accuracy',  
        cv=10,  
        verbose=1,
        n_jobs=-1  
    )


    grid_search.fit(X_tr, Y_tr)


    best_model = grid_search.best_estimator_
    best_model.fit(X_tr, Y_tr)


    nb_pred = best_model.predict(X_te)
    ConfusionMatrixDisplay.from_estimator(estimator=best_model, X=X_te, y=Y_te)

    print("Classification Report: Naive Bayes (GaussianNB)")
    report = classification_report(Y_te, nb_pred, digits=2)
    print(report)
    return best_model


def NaiveBayesP(X_tr, Y_tr, X_te, Y_te, quick=False):
    param_grid = {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]  
    }

    
    nb_model = GaussianNB()

    if quick:
        
        best_model = nb_model
        best_model.fit(X_tr, Y_tr)

       
        nb_pred = best_model.predict(X_te)
        nb_prob = best_model.predict_proba(X_te)  
        for y, pred, prob in zip(Y_te, nb_pred, nb_prob):
            print(f"True: {y}, Pred: {pred}, Probabilities: {prob}")


        print("Classification Report: Naive Bayes (GaussianNB)")
        report = classification_report(Y_te, nb_pred, digits=2)
        print(report)
        return best_model

    
    grid_search = GridSearchCV(
        estimator=nb_model,
        param_grid=param_grid,
        scoring='accuracy',  
        cv=10,  
        verbose=1,
        n_jobs=-1  
    )

   
    grid_search.fit(X_tr, Y_tr)

   
    best_model = grid_search.best_estimator_
    best_model.fit(X_tr, Y_tr)

   
    nb_pred = best_model.predict(X_te)
    nb_prob = best_model.predict_proba(X_te)  # Get predicted probabilities

    for y, pred, prob in zip(Y_te, nb_pred, nb_prob):
        print(f"True: {y}, Pred: {pred}, Probabilities: {prob}")

    ConfusionMatrixDisplay.from_estimator(estimator=best_model, X=X_te, y=Y_te)

    print("Classification Report: Naive Bayes (GaussianNB)")
    report = classification_report(Y_te, nb_pred, digits=2)
    print(report)
    return best_model




def SVM(X_tr, Y_tr, X_te, Y_te, quick=False):
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto'],
        'degree': [3, 5],
        'random_state': [None,42]
    }

    # Create the model
    svc_model = SVC()

    if quick:
        best_model = SVC(kernel='rbf')
        best_model.fit(X_tr, Y_tr)

        svc_pred = best_model.predict(X_te)

        ConfusionMatrixDisplay.from_estimator(estimator=best_model, X=X_te, y=Y_te)

        print("Classification Report: SVC")
        report = classification_report(Y_te, svc_pred, digits=2)
        print(report)
        return best_model

    grid_search = GridSearchCV(
        estimator=svc_model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=10,
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_tr, Y_tr)

    best_model = grid_search.best_estimator_
    best_model.fit(X_tr, Y_tr)

    svc_pred = best_model.predict(X_te)

    ConfusionMatrixDisplay.from_estimator(estimator=best_model, X=X_te, y=Y_te)

    print("Classification Report: SVC")
    report = classification_report(Y_te, svc_pred, digits=2)
    print(report)
    return best_model




def SVMP(X_tr, Y_tr, X_te, Y_te, quick=False):
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto'],
        'degree': [3, 5],
        'random_state':[None,42]
    }

    # Create the model with probability estimates enabled
    svcp_model = SVC(probability=True )

    if quick:
        best_model = SVC(probability=True,kernel='rbf')
        best_model.fit(X_tr, Y_tr)

        svcp_pred = best_model.predict(X_te)
        svcp_prob = best_model.predict_proba(X_te)

        for y, pred, prob in zip(Y_te, svcp_pred, svcp_prob):
            print(f"True: {y}, Pred: {pred}, Probabilities: {prob}")

        svcp_prob = [max(a) for a in svcp_prob]

        print("Classification Report: SVC (probability)")
        report = classification_report(Y_te, svcp_pred, digits=2)
        print(report)
        return best_model

    grid_search = GridSearchCV(
        estimator=svcp_model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=10,
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_tr, Y_tr)

    best_model = grid_search.best_estimator_
    best_model.fit(X_tr, Y_tr)

    svcp_pred = best_model.predict(X_te)

    ConfusionMatrixDisplay.from_estimator(estimator=best_model, X=X_te, y=Y_te)

    print("Classification Report: SVCP")
    report = classification_report(Y_te, svcp_pred, digits=2)
    print(report)
    return best_model
