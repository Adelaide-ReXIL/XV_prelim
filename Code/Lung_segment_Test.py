

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import pandas as pd
    from Triangle_Test import plot_3d_points,make_graphs_plots
    import os
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score


@app.function
def centralise(data):
    for x in ['x','y','z']:
        data[x]=data[x]-data[x].min()
    return data


@app.function
def rank(data):
    for x in ['x','y','z']:
        data[x]=data[x].rank(method='dense')
    return data


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    data=pd.read_csv('../Datasets/Synth/real_1.csv')
    data.columns=['SV','x','y','z']
    return (data,)


@app.cell
def _(data):
    plot_3d_points(data)
    return


@app.cell
def _(data):
    centrelaised=centralise(data.copy())
    return (centrelaised,)


@app.cell
def _(centrelaised):
    plot_3d_points(centrelaised)
    return


@app.cell
def _(centrelaised, data):
    ranked=rank(data.copy())
    plot_3d_points(centrelaised)
    return


@app.cell
def _(data):
    rank(data)
    return


@app.cell
def _(data):
    segments=data.copy()
    x_mean = segments['x'].mean()
    y_mean = segments['y'].mean()
    z_mean = segments['z'].mean()

    segments['segment'] = (
        (segments['x'] > x_mean).astype(int) * 1 +
        (segments['y'] > y_mean).astype(int) * 2 +
        (segments['z'] > z_mean).astype(int) * 4
    )
    return (segments,)


@app.cell
def _(segments):
    temp=segments.copy()
    temp['SV']=temp['segment']
    plot_3d_points(temp)
    return


@app.cell
def _(segments):
    grouped = segments.groupby('segment')
    for k,v in grouped:
        print(v)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Lobar Segmentation
        Trying to make a clustering model that segmetn the lungs into 5 lobar regions.
        """
    )
    return


@app.function
def norm(data):
    for x in ['x', 'y', 'z']:
        data[x] = (data[x] - data[x].min()) / (data[x].max() - data[x].min())
    return data


@app.function
def clean_lung(lung):
    lung=lung[lung['Frame']==0]
    lung=lung.drop(columns=['Frame'])
    lung=lung[["divergence.0, 0",'position.x','position.y','position.z','lobe']]
    lung.columns=['SV','x','y','z','lobe']
    rank(lung)
    norm(lung)
    return lung


@app.function
def get_lung_and_lobes(name=''):
    return get_insp_exp(name,'INSP'),get_insp_exp(name,'EXP')


@app.function
def get_insp_exp(name,mode):
    lung=pd.DataFrame()
    drop='-'.join(name.split('-')[:-1])
    path=f'../Datasets/XV Clinical Data/{name}/{name}-{mode}-LOBAR/'
    new_path=f'../Datasets/XV Clinical Data/{name}/{drop}-{mode}-LOBAR/'

    new_path_sup=f'../Datasets/XV Clinical Data/{name}/{drop}-SUPINE-{mode}-LOBAR/'
    lobes=['LLL',"LUL","RLL",'RML',"RUL"]



    for i,l in enumerate(lobes):

        try:
            lobe=pd.read_csv(path+l+'/'+drop+'-'+mode+'-'+l+'-final.csv')
            lobe['lobe']=i
            lung=pd.concat([lung,lobe])
            continue
        except FileNotFoundError as e:
            ...

        try:
            lobe=pd.read_csv(new_path+l+'/'+drop+'-'+mode+'_final.csv')
            lobe['lobe']=i
            lung=pd.concat([lung,lobe])
            continue
        except FileNotFoundError as e:
            ...

        lobe=pd.read_csv(new_path_sup+l+'/'+drop+'-'+mode+'_final.csv')
        lobe['lobe']=i
        lung=pd.concat([lung,lobe])




    return clean_lung(lung)


@app.cell
def _():
    insp,exp=get_lung_and_lobes('WCH-CF-10003-20230317')
    return exp, insp


@app.cell
def _(insp):

    insp['SV']=insp['lobe']
    plot_3d_points(insp)
    return


@app.cell
def _(exp):

    exp['SV']=exp['lobe']
    plot_3d_points(exp)
    return


@app.cell
def _():
    csv_dir = '../Datasets/XV Clinical Data'
    csv_files = [fold for fold in os.listdir(csv_dir) if os.path.isdir(os.path.join(csv_dir, fold)) and 'WCH' in fold]

    csv_files
    return (csv_files,)


@app.cell
def _(csv_files):
    all_ins_outs=[]

    for file in csv_files:
        if file in ['WCH-CF-10004-20230331','WCH-CF-10023-20240726']:
            continue
        a,b=get_lung_and_lobes(file)
        all_ins_outs.append(a)
        all_ins_outs.append(b)



    return (all_ins_outs,)


@app.cell
def _(all_ins_outs):
    combined=pd.DataFrame()
    for d in all_ins_outs:
        combined=pd.concat([combined,d])
    combined=combined.drop(columns=['SV'])
    return (combined,)


@app.cell
def _(combined):
    Y=combined['lobe']
    X=combined.drop(columns=['lobe'])
    return X, Y


@app.cell
def _(X, Y):


    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

    param_grid = {
        'n_neighbors': [65,75,95],
        'leaf_size':[5,10,20,30,50]

    }

    model = GridSearchCV(KNeighborsClassifier(p=2,leaf_size=5), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    model.fit(X_train, y_train)

    print("Best Params:", model.best_params_)
    print("Best CV Score:", model.best_score_)

    y_pred = model.predict(X_test)
    print("Test Accuracy:", accuracy_score(y_test, y_pred))

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return (model,)


@app.cell
def _(data):
    rank(data)
    norm(data)
    return


@app.cell
def _(data, model):
    x=data.drop(columns=['SV'])
    y=model.predict(x)
    return (y,)


@app.cell
def _(data):
    data
    return


@app.cell
def _(data, y):
    data['SV']=y
    plot_3d_points(data)
    return


@app.cell
def _():
    test2=pd.read_csv('../Datasets/Rat Sterile Bead Study/csv/baseline/3754.Phe508.specificVentilation.csv')
    test2.columns=['SV','x','y','z']
    rank(test2) 
    norm(test2)
    return (test2,)


@app.cell
def _(model, test2):
    x_t=test2.drop(columns=['SV'])
    y_t=model.predict(x_t)
    return (y_t,)


@app.cell
def _(test2, y_t):
    test2['SV']=y_t
    plot_3d_points(test2)
    return


@app.cell
def _(test2):
    test2
    return


@app.cell
def _(model):
    import pickle

    with open('model.pkl', 'wb') as f:
        pickle.dump(model.best_estimator_, f)
    return


@app.cell
def _(insp):
    plot_3d_points(insp)
    return


@app.cell
def _(insp, model):
    X_i=insp.copy()[['x','y','z']]
    y_i=model.predict(X_i)

    X_i['SV']=y_i
    plot_3d_points(X_i)
    return


if __name__ == "__main__":
    app.run()
