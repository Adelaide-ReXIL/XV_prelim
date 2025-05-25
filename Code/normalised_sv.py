

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")

with app.setup:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from Lung_segment_Test import norm,rank
    import scipy.optimize as optimize
    import scipy.ndimage as ndi
    import matplotlib.pyplot as plt
    # from Triangle_Test import plot_3d_points
    import os
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from scipy.interpolate import griddata
    from scipy.ndimage import distance_transform_edt
    from sklearn.preprocessing import normalize
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_squared_error, r2_score
    from lung_graph import lung_plot


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.function
def plot_3d_points(df,vmin=-0.22,vmax=0.89):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(df['x'], df['y'], df['z'], c=df['SV'], cmap='RdYlGn', s=10, vmax=vmax, vmin=vmin)
    fig.colorbar(sc, ax=ax, label='SV')
    return ax


@app.function
def norm_super(data):
    for c in ['x', 'y', 'z']:
        min_val = data[c].min()
        max_val = data[c].max()
        if max_val != min_val:
            data[c] = (data[c] - min_val) / (max_val - min_val)
        else:
            data[c] = 0
    return data


@app.function
def rotate_lung_data(df, show_plots=False):
    """
    Rotate lung data to optimal orientation

    Parameters:
    df (pandas.DataFrame): DataFrame with columns [specific_ventilation, x, y, z]
    show_plots (bool): Whether to display diagnostic plots

    Returns:
    pandas.DataFrame: DataFrame with rotated coordinates
    """
    # Convert to numpy array expected by rotation functions
    data = df.values

    # Find optimal rotation angle
    angle = find_optimal_angle(data, show_plots)

    # Apply rotation
    xs, ys, zs, vals = rotate_points(data, angle)

    # Create output DataFrame with rotated coordinates
    rotated_df = pd.DataFrame({
        'SV': vals,
        'x': xs,
        'y': ys,
        'z': zs,
    })

    return rotated_df


@app.function
def rotate_points(data, angle):
    """Rotate x,y coordinates by given angle in degrees"""
    xs = data[:,1]  # x coordinates in column 1
    ys = data[:,2]  # y coordinates in column 2
    zs = data[:,3]  # z coordinates in column 3
    vals = data[:,0]  # values in column 0 (specific ventilation)

    # Convert angle to radians and apply rotation
    theta = angle * np.pi/180
    xnew = np.cos(theta)*xs - np.sin(theta)*ys
    ynew = np.sin(theta)*xs + np.cos(theta)*ys

    return xnew, ynew, zs, vals


@app.function
def quick_gauss(x, a, b, c):
    """Gaussian function for curve fitting"""
    return abs(a) * np.exp(-1 * ((x-c)/b)**2)


@app.function
def periodic_function(angle, amp, phase, height_shift):
    """Periodic function for fitting angle data"""
    theta = (angle - phase) * np.pi/180
    return abs(amp * (np.sin(theta))) + height_shift


@app.function
def find_optimal_angle(data, show_plots=False):
    """Find optimal angle for lung orientation"""
    # Test a range of angles
    angles_to_try = np.linspace(0, 360, 200)
    widths = []
    valid_angles = []

    # For each angle, measure "narrowness"
    for angle in angles_to_try:
        try:
            # Rotate data by this angle
            xs, ys, zs, vals = rotate_points(data, angle)

            # Use most of the data for analysis
            z_threshold = np.percentile(zs, 95)
            y_filtered = [ys[i] for i in range(len(zs)) if zs[i] < z_threshold]

            # Get histogram of y values (lung profile from side)
            hist, edges = np.histogram(y_filtered, 50)
            hist_centers = (edges[:-1] + edges[1:]) / 2

            # Find valid histogram bins
            valid_indices = hist > 1
            valid_x = hist_centers[valid_indices]
            valid_y = hist[valid_indices]

            if len(valid_x) < 3:
                continue

            # Fit gaussian to profile
            params, _ = optimize.curve_fit(
                quick_gauss, valid_x, valid_y,
                p0=[np.max(valid_y), 1, 0], maxfev=10000
            )

            # Width parameter (standard deviation)
            width = -abs(params[1])  # Negative so peaks become valleys

            widths.append(width)
            valid_angles.append(angle)

        except:
            # Skip problematic angles
            continue

    # Apply median filter to smooth measurements
    filtered_widths = ndi.median_filter(widths, size=5)

    try:
        # Try to fit with initial guess of phase=45 degrees
        params1, _ = optimize.curve_fit(
            periodic_function, valid_angles, filtered_widths,
            p0=[np.ptp(filtered_widths)/2, 45, np.min(filtered_widths)],
            maxfev=10000
        )

        # Try again with initial guess of phase=-45 degrees
        params2, _ = optimize.curve_fit(
            periodic_function, valid_angles, filtered_widths,
            p0=[np.ptp(filtered_widths)/2, -45, np.min(filtered_widths)],
            maxfev=10000
        )

        # Choose the better fit
        residuals1 = np.sum((periodic_function(valid_angles, *params1) - filtered_widths)**2)
        residuals2 = np.sum((periodic_function(valid_angles, *params2) - filtered_widths)**2)

        params = params1 if residuals1 < residuals2 else params2

        # Normalize angle to range [-90, 90]
        optimal_angle = params[1] % 180 - 90

        if show_plots:
            # Plot the angle optimization
            plt.figure(figsize=(10, 6))
            plt.plot(valid_angles, filtered_widths, 'o', alpha=0.5, label='Width measurements')

            # Plot the fitted curve
            x_fit = np.linspace(0, 360, 500)
            y_fit = periodic_function(x_fit, *params)
            plt.plot(x_fit, y_fit, 'r-', label='Fitted curve')

            plt.axvline(optimal_angle, color='g', linestyle='--', label=f'Optimal angle: {optimal_angle:.2f}°')
            plt.title('Finding Optimal Rotation Angle')
            plt.xlabel('Angle (degrees)')
            plt.ylabel('Width parameter')
            plt.legend()

            # Plot the actual rotated data
            plt.figure(figsize=(10, 10))
            rotated_x, rotated_y, rotated_z, _ = rotate_points(data, optimal_angle)
            plt.scatter(rotated_x, rotated_y, c=rotated_z, alpha=0.3, s=2, cmap='RdYlGn')
            plt.colorbar(label='Z position')
            plt.title(f'Lung Data Rotated by {optimal_angle:.2f}°')
            plt.xlabel('X position')
            plt.ylabel('Y position')
            plt.axis('equal')
            plt.show()

        return optimal_angle

    except:
        # If curve fitting fails, return 0 (no rotation)
        print("Warning: Failed to find optimal angle, using 0°")
        return 0


@app.cell
def _():
    ddata=pd.read_csv('../Datasets/Rat Sterile Bead Study/csv/baseline/3761.WT.specificVentilation.csv')
    ddata.columns=['SV','x','y','z']
    return (ddata,)


@app.cell
def _(ddata):
    plot_3d_points(ddata)
    return


@app.cell
def _(ddata):
    rot=rotate_lung_data(ddata)
    return (rot,)


@app.cell
def _(rot):
    rot
    return


@app.cell
def _(rot):
    plot_3d_points(rot)
    return


@app.cell
def _(ddata):
    data=rotate_lung_data(ddata)
    rank(data)
    norm(data)
    return (data,)


@app.cell
def _(data):
    plot_3d_points(data)
    return


@app.function
def make_voxel(data):


    data = np.array(data.values) # shape (N, 4): [metric, x, y, z]


    D, H, W = 32, 32, 32

    x_idx = np.clip((data[:,1] * (W - 1)).astype(int), 0, W - 1)
    y_idx = np.clip((data[:,2] * (H - 1)).astype(int), 0, H - 1)
    z_idx = np.clip((data[:,3] * (D - 1)).astype(int), 0, D - 1)

    volume = np.full((1, D, H, W), np.nan, dtype=np.float32)

    for i in range(len(data)):
        v = volume[0, x_idx[i], y_idx[i], z_idx[i]]
        volume[0, x_idx[i], y_idx[i], z_idx[i]] = data[i, 0] if np.isnan(v) else (data[i, 0] + v) / 2
    return volume


@app.cell
def _(data):
    volume=make_voxel(data=data)
    return (volume,)


@app.function
def plot_volume_slices(volume, axis=1, num_slices=6):
    # volume: (1, D, H, W)
    v = volume[0]  # drop channel dim
    dims = v.shape

    indices = np.linspace(0, dims[axis] - 1, num_slices).astype(int)

    fig, axes = plt.subplots(1, num_slices, figsize=(15, 3))
    for i, idx in enumerate(indices):
        if axis == 0:
            img = v[idx, :, :]
        elif axis == 1:
            img = v[:, idx, :]
        else:
            img = v[:, :, idx]
        axes[i].imshow(img, cmap='RdYlGn')
        axes[i].set_title(f"{['Z','Y','X'][axis]}={idx}")
        axes[i].axis('off')
    plt.tight_layout()
    return plt.gca()


@app.cell
def _(volume):
    plot_volume_slices(volume,axis=2)
    return


@app.function
def plot_voxels(volume, threshold=-0.6):
    # volume: shape (1, D, H, W)
    v = volume[0]
    filled = v > threshold
    D, H, W = filled.shape
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(filled, facecolors='cyan', edgecolor='k')
    ax.set(xlabel='X', ylabel='Y', zlabel='Z')

    return plt.gca()


@app.function
def plot_voxels_color(volume, threshold=-0.6, cmap='RdYlGn'):
    v = volume[0]
    mask = (v > threshold) & ~np.isnan(v)

    normed_vals = (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v) + 1e-8)
    normed_vals = np.clip(normed_vals, 0, 1)

    colors = cm.get_cmap(cmap)(normed_vals)

    facecolors = np.zeros((*v.shape, 4))  # RGBA
    facecolors[mask] = colors[mask]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(mask, facecolors=facecolors, edgecolor='k')
    ax.set(xlabel='X', ylabel='Y', zlabel='Z')
    return ax


@app.cell
def _(volume):
    plot_voxels(volume)
    return


@app.cell
def _(volume):
    plot_voxels_color(volume) 
    return


@app.cell
def _(volume):
    np.isnan(volume[0, :, :, 3]).mean() 
    return


@app.function
def interpolate_volume(volume):
    v = volume[0]
    D, H, W = v.shape

    grid_z, grid_y, grid_x = np.mgrid[0:D, 0:H, 0:W]
    valid = ~np.isnan(v)

    points = np.vstack((grid_z[valid], grid_y[valid], grid_x[valid])).T
    values = v[valid]

    full_coords = np.vstack((grid_z.ravel(), grid_y.ravel(), grid_x.ravel())).T
    filled = griddata(points, values, full_coords, method='linear')
    filled = filled.reshape((D, H, W))

    return filled[np.newaxis]


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Got side tracked make model for healthy lung now <br />""")
    return


@app.cell
def _():
    csv_dir = '../Datasets/Rat Sterile Bead Study/csv/baseline'
    files = [csv_dir+'/'+fold for fold in os.listdir(csv_dir) if  'WT' in fold]
    csv_files=files
    csv_dir = '../Datasets/Rat Sterile Bead Study/csv/Control'
    files = [csv_dir+'/'+fold for fold in os.listdir(csv_dir) if  'WT' in fold]
    csv_files.extend(files)

    csv_files

    return (csv_files,)


@app.cell
def _(csv_files, data):
    combined=pd.DataFrame()
    for file in csv_files:
        temp=pd.read_csv(file)
        temp.columns=['SV','x','y','z']
        temp=rotate_lung_data(data)
        rank(data)
        norm(data)
        combined=pd.concat([combined,data],ignore_index=True)


    combined



    return (combined,)


@app.cell
def _():
    scaler=MinMaxScaler()
    return (scaler,)


@app.cell
def _(combined):
    Y=combined['SV']
    X=combined.drop(columns=['SV'])
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    return X_test, X_train, y_test, y_train


@app.cell(disabled=True)
def _(X_test, X_train, scaler, y_test, y_train):


    from xgboost import XGBRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.svm import SVR
    from sklearn.linear_model import Ridge

    # 1. KNN Regressor
    pipeline_knn = Pipeline([
        ('scaler', scaler),
        ('regressor', KNeighborsRegressor())
    ])
    param_grid_knn = {
        'regressor__n_neighbors': [65, 75, 95],
        'regressor__leaf_size': [5, 10, 20, 30, 50]
    }
    model_knn = GridSearchCV(pipeline_knn, param_grid_knn, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    model_knn.fit(X_train, y_train)
    y_pred_knn = model_knn.predict(X_test)
    print('--- KNN Regressor ---')
    print("Best Params:", model_knn.best_params_)
    print("Best CV Score (neg MSE):", model_knn.best_score_)
    print("Test MSE:", mean_squared_error(y_test, y_pred_knn))
    print("R² Score:", r2_score(y_test, y_pred_knn))

    # 2. Random Forest Regressor
    pipeline_rf = Pipeline([
        ('scaler', scaler),
        ('regressor', RandomForestRegressor())
    ])
    param_grid_rf = {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [None, 10, 20],
        'regressor__min_samples_split': [2, 5],
        'regressor__min_samples_leaf': [1, 2]
    }
    model_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    model_rf.fit(X_train, y_train)
    y_pred_rf = model_rf.predict(X_test)
    print('--- Random Forest Regressor ---')
    print("Best Params:", model_rf.best_params_)
    print("Best CV Score (neg MSE):", model_rf.best_score_)
    print("Test MSE:", mean_squared_error(y_test, y_pred_rf))
    print("R² Score:", r2_score(y_test, y_pred_rf))

    # # 3. Gradient Boosting Regressor
    # pipeline_gbr = Pipeline([
    #     ('scaler', scaler),
    #     ('regressor', GradientBoostingRegressor())
    # ])
    # param_grid_gbr = {
    #     'regressor__n_estimators': [100, 200],
    #     'regressor__learning_rate': [0.05, 0.1],
    #     'regressor__max_depth': [3, 5],
    #     'regressor__subsample': [0.8, 1.0]
    # }
    # model_gbr = GridSearchCV(pipeline_gbr, param_grid_gbr, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    # model_gbr.fit(X_train, y_train)
    # y_pred_gbr = model_gbr.predict(X_test)
    # print('--- Gradient Boosting Regressor ---')
    # print("Best Params:", model_gbr.best_params_)
    # print("Best CV Score (neg MSE):", model_gbr.best_score_)
    # print("Test MSE:", mean_squared_error(y_test, y_pred_gbr))
    # print("R² Score:", r2_score(y_test, y_pred_gbr))

    # # 4. SVR
    # pipeline_svr = Pipeline([
    #     ('scaler', scaler),
    #     ('regressor', SVR())
    # ])
    # param_grid_svr = {
    #     'regressor__C': [0.1, 1, 10],
    #     'regressor__epsilon': [0.01, 0.1],
    #     'regressor__kernel': ['rbf', 'linear']
    # }
    # model_svr = GridSearchCV(pipeline_svr, param_grid_svr, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    # model_svr.fit(X_train, y_train)
    # y_pred_svr = model_svr.predict(X_test)
    # print('--- SVR ---')
    # print("Best Params:", model_svr.best_params_)
    # print("Best CV Score (neg MSE):", model_svr.best_score_)
    # print("Test MSE:", mean_squared_error(y_test, y_pred_svr))
    # print("R² Score:", r2_score(y_test, y_pred_svr))

    # # 5. Ridge Regression
    # pipeline_ridge = Pipeline([
    #     ('scaler', scaler),
    #     ('regressor', Ridge())
    # ])
    # param_grid_ridge = {
    #     'regressor__alpha': [0.1, 1.0, 10.0]
    # }
    # model_ridge = GridSearchCV(pipeline_ridge, param_grid_ridge, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    # model_ridge.fit(X_train, y_train)
    # y_pred_ridge = model_ridge.predict(X_test)
    # print('--- Ridge Regression ---')
    # print("Best Params:", model_ridge.best_params_)
    # print("Best CV Score (neg MSE):", model_ridge.best_score_)
    # print("Test MSE:", mean_squared_error(y_test, y_pred_ridge))
    # print("R² Score:", r2_score(y_test, y_pred_ridge))


    pipeline_xgb = Pipeline([
        ('scaler', scaler),
        ('regressor', XGBRegressor(objective='reg:squarederror', verbosity=0, n_jobs=-1))
    ])
    param_grid_xgb = {
        'regressor__n_estimators': [100, 200],
        'regressor__learning_rate': [0.05, 0.1],
        'regressor__max_depth': [3, 5],
        'regressor__subsample': [0.8, 1.0],
        'regressor__colsample_bytree': [0.8, 1.0]
    }
    model_xgb = GridSearchCV(pipeline_xgb, param_grid_xgb, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    model_xgb.fit(X_train, y_train)
    y_pred_xgb = model_xgb.predict(X_test)
    print('--- XGBoost Regressor ---')
    print("Best Params:", model_xgb.best_params_)
    print("Best CV Score (neg MSE):", model_xgb.best_score_)
    print("Test MSE:", mean_squared_error(y_test, y_pred_xgb))
    print("R² Score:", r2_score(y_test, y_pred_xgb))
    return


@app.function
def get_diff(path='',model=None,df=None):
        test=None
        if path=='':
            test=df
        else:
            test=pd.read_csv(path)
            test=rotate_lung_data(test)
        rank(test)
        norm(test)

        # test_1=pd.DataFrame(scaler.transform(test))
        test_2=test.copy()
        pred=model.predict(test.drop(columns=['SV']))
        print(np.sqrt(mean_squared_error(pred,test['SV'])))
        test_2["SV"]=pred
        test_3=test_2.copy() 
        test_3['SV']=test['SV']**2-test_2['SV']**2
        return test,test_2,test_3,plot_3d_points(test),plot_3d_points(test_2),plot_3d_points(test_3,-0.05,0.05)


@app.cell
def _():
    import pickle
    with open('model_segment.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model_normal.pkl', 'rb') as f:
        model_diff = pickle.load(f)

    return model, model_diff


@app.cell
def _(model_diff):
    r1=get_diff('../Datasets/Rat PA Study/csv/5708.KO.PA63.pp2.specificVentilation.csv',model_diff)
    r1
    return (r1,)


@app.cell
def _(model, r1):
    lung_plot(df=r1[2],model=model,vmax=0.08,vmin=-0.08)
    return


@app.cell
def _(mo):
    mo.md(r"""# Examples""")
    return


@app.cell
def _(mo):
    mo.md(r"""Example:  5766.WT.PA63""")
    return


@app.cell
def _(model_diff):
    r2=get_diff('../Datasets/Rat PA Study/csv/5766.WT.PA63.pp2.specificVentilation.csv',model_diff)
    r2
    return (r2,)


@app.cell
def _(model, r2):
    lung_plot(df=r2[2],model=model,vmax=0.08,vmin=-0.08)
    return


@app.cell
def _(mo):
    mo.md(r"""Example: 4570.KO.beads""")
    return


@app.cell
def _(model_diff):
    r3=get_diff('../Datasets/Rat Sterile Bead Study/csv/post_beads/4570.KO.beads.specificVentilation.csv',model_diff)
    r3
    return (r3,)


@app.cell
def _(model, r3):
    lung_plot(df=r3[2],model=model,vmax=0.08,vmin=-0.08)
    return


@app.cell
def _(mo):
    mo.md(r"""Example: 4869.WT.BEADS""")
    return


@app.cell
def _(model_diff):
    r4=get_diff('../Datasets/Rat Sterile Bead Study/csv/post_beads/4869.WT.BEADS.specificVentilation.csv',model_diff)
    r4
    return (r4,)


@app.cell
def _(model, r4):
    lung_plot(df=r4[2],model=model,vmax=0.08,vmin=-0.08)
    return


@app.cell
def _(r4):
    tester=r4[2].copy()
    tester['SV']=tester['segments']
    plot_3d_points(tester,vmax=6,vmin=0)
    return


@app.cell
def _(mo):
    mo.md(r"""Example: 3757.KO""")
    return


@app.cell
def _(model_diff):
    r5=get_diff('../Datasets/Rat Sterile Bead Study/csv/baseline/3757.KO.specificVentilation.csv',model_diff)
    r5
    return (r5,)


@app.cell
def _(model, r5):
    lung_plot(df=r5[2],model=model,vmax=0.08,vmin=-0.08)
    return


@app.cell
def _(mo):
    mo.md(r"""Example: 4802.WT.ETI.control""")
    return


@app.cell
def _(model_diff):
    r6=get_diff("../Datasets/Rat Sterile Bead Study/csv/Control/4802.WT.ETI.control.specificVentilation.csv",model_diff)
    r6
    return (r6,)


@app.cell
def _(model, r6):
    lung_plot(df=r6[2],model=model,vmax=0.08,vmin=-0.08)
    return


@app.cell
def _(mo):
    mo.md(r"""## No. 17""")
    return


@app.cell
def _(model):
    c17=pd.read_csv('../Datasets/XV Clinical Data/WCH-CF-10017-20240517/WCH-CF-10017-INSP/WCH-CF-10017-INSP_final.csv')
    c17.columns=['Frame','SV','x','y','z']
    c17_res=[]
    for i_17 in range(7):
        temp_17=c17[c17['Frame']==i_17]
        temp_17=temp_17.drop(columns=['Frame'])
        c17_res.append(lung_plot(df=temp_17,model=model,vmin=temp_17['SV'].min(),vmax=temp_17['SV'].max()))

    return (c17_res,)


@app.cell
def _(c17_res):
    c17_res[0]
    return


@app.cell
def _(mo):
    mo.md(r"""## No. 3""")
    return


@app.cell
def _(model):
    c3=pd.read_csv('../Datasets/XV Clinical Data/WCH-CF-10003-20230317/WCH-CF-10003-20230317-INSP/WCH-CF-10003-INSP_final.csv')
    c3.columns=['Frame','SV','x','y','z']
    c3_res=[]
    for i_3 in range(7):
        temp_3=c3[c3['Frame']==i_3]
        temp_3=temp_3.drop(columns=['Frame'])
        c3_res.append(lung_plot(df=temp_3,model=model,vmin=temp_3['SV'].min(),vmax=temp_3['SV'].max()))
    return (c3_res,)


@app.cell
def _(c3_res):
    c3_res[6]
    return


@app.cell
def _(mo):
    mo.md(r"""## No. 6""")
    return


@app.cell
def _(model):
    c6=pd.read_csv('../Datasets/XV Clinical Data/WCH-CF-10006-20230519/WCH-CF-10006-SUPINE-INSP/WCH-CF-10006-INSP_final.csv')
    c6.columns=['Frame','SV','x','y','z']
    c6_res=[]
    for i_6 in range(7):
        temp_6=c6[c6['Frame']==i_6]
        temp_6=temp_6.drop(columns=['Frame'])
        c6_res.append(lung_plot(df=temp_6,model=model,vmin=temp_6['SV'].min(),vmax=temp_6['SV'].max()))

    return (c6_res,)


@app.cell
def _(c6_res):
    c6_res[0]
    return


if __name__ == "__main__":
    app.run()
