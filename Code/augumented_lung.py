

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")

with app.setup:

    import numpy as np
    import pandas as pd
    from Lung_segment_Test import norm
    from Triangle_Test import plot_3d_points


@app.cell
def _():
    import marimo as mo
    return


@app.function
def aug_lung(data, num_aug):
    all_aug = [data]

    for _ in range(num_aug):
        aug_copy = data.copy()

        keep_mask = np.random.rand(len(aug_copy)) <= 1/3
        aug_copy = aug_copy[keep_mask]

        noise = {
            col: 1 + np.random.uniform(-1/30, 1/30, size=len(aug_copy))
            for col in ["x", "y", "z"]
        }
        noise["SV"] = 1 + np.random.uniform(-0.001, 0.001, size=len(aug_copy))

        for col in ["SV", "x", "y", "z"]:
            aug_copy[col] *= noise[col]

        all_aug.append(aug_copy.reset_index(drop=True))
        data = pd.concat(all_aug, ignore_index=True)
        all_aug=[data]

    norm(data)
    return data


@app.cell
def _():
    data=pd.read_csv('../Datasets/XV Clinical Data/WCH-CF-10024-20240802/WCH-CF-10024-EXP/WCH-CF-10024-EXP_final.csv')
    data=data[data['Frame']==6]
    data=data.drop(columns=['Frame'])
    data.columns=['SV','x','y','z']
    norm(data)
    data
    return (data,)


@app.cell
def _(data):
    plot_3d_points(data)

    return


@app.cell
def _(data):
    aug=aug_lung(data,int((10000/len(data))))

    print(len(aug),len(data))
    plot_3d_points(data),plot_3d_points(aug)

    return (aug,)


@app.cell
def _(data):


    import matplotlib.pyplot as plt

    plt.scatter(data['x'],data['y'])
    return (plt,)


@app.cell
def _(aug, plt):

    plt.scatter(aug['x'],aug['y'])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
