import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")

with app.setup:
    import numpy as np 
    from Lung_segment_Test import get_lung_and_lobes,rank,norm
    import pickle
    import pandas as pd 
    from Triangle_Test  import make_graphs_plots,get_graphs
    from sklearn.neighbors import KNeighborsClassifier
    import matplotlib.pyplot as plt


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    with open('model_segment.pkl', 'rb') as f:
        model = pickle.load(f)
    return (model,)


@app.function
def plot_3d_points(df,ax,vmin=-0.22,vmax=0.89):

    ax.scatter(df['x'], df['y'], df['z'], c=df['SV'], cmap='RdYlGn', s=10,vmin=vmin,vmax=vmax)
    return ax


@app.cell
def _():
    return


@app.function
def add_lobes(data,model:KNeighborsClassifier):
    X=data[['x','y','z']]
    lobe=model.predict(X)
    data['segments']=lobe


@app.cell
def _():

    data=pd.read_csv('../Datasets/Synth/real_1.csv')
    data.columns=['SV','x','y','z']
    rank(data)
    norm(data)
    data
    return (data,)


@app.cell
def _(data):
    ax=plt.figure(figsize=(12, 6),projection='3d')
    plot_3d_points(data,ax)
    return


@app.cell
def _(data, model):
    add_lobes(data,model)
    data
    return


@app.cell
def _(data):
    temp=data.copy()
    temp['SV']=temp['segments']
    plot_3d_points(temp)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Making graph from Lungs""")
    return


@app.cell
def _():
    return


@app.function
def make_lobe_graphs(lung):
    lobes=[]
    for lobe in range(0,5):
        temp=lung[lung['segments']==lobe].drop(columns=['segments'])
        if temp.empty:
            lobes.append([])
            continue
        df=get_graphs('',temp.copy())
        lobes.append(df)
    return lobes


@app.cell
def _(data):
    lobes=make_lobe_graphs(data)
    return (lobes,)


@app.cell
def _(lobes):
    lobes

    return


@app.cell
def _(lobes):
    l=lobes[1]
    return (l,)


@app.cell
def _(data, l):
    plot_3d_points(data[data['segments']==1])
    plt.scatter(l['y'],l['x'],s=l['z']*50,c=l['SV'])
    return


@app.cell
def _(data, lobes):
    def _():
        for i in range(5):
            fig = plt.figure(figsize=(12, 6))

            ax1 = fig.add_subplot(1, 2, 1, projection='3d')
            plot_3d_points(data[data['segments'] == i], ax1)

            ax2 = fig.add_subplot(1, 2, 2)
            l=lobes[i]
            ax2.scatter(l['y'], l['x'], s=l['z'] * 50, c=l['SV'])

            plt.tight_layout()
        return plt.show()


    _()
    return


@app.cell
def _():
    return


@app.function
def lung_plot(path='',df=None,model=None,vmin=-0.22,vmax=0.89):
    data=None
    if path!='':
        data=pd.read_csv(path)
    else:
        data=df



    data.columns=['SV','x','y','z']
    rank(data)
    norm(data)
    add_lobes(data,model)
    lobes=make_lobe_graphs(data)
    plts=[]
    for i in range(5):
        fig = plt.figure(figsize=(12, 6))

        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.set_facecolor('white')
        ax1.set_axis_off()
        
        



        ax2 = fig.add_subplot(1, 2, 2) 
        l=lobes[i]
        segment=data[data['segments'] == i].copy()
        plot_3d_points(segment, ax1,vmin,vmax)
        l['s']=(l['s'] - l['s'].min()) / (l['s'].max() - l['s'].min())
        scale= (1 - l['s']) * 30 + 10
        sc = ax2.scatter(l['y'], l['x'], s=scale, c=l['SV'], alpha=(l['s'] + 0.4).clip(upper=1), cmap='RdYlGn',vmax=vmax,vmin=vmin)
        cbar = plt.colorbar(sc, ax=ax2)
        cbar.set_label('SV (0 to 0.5)')
        plt.tight_layout()
        plts.append(fig)

    return plts,lobes


@app.cell
def _(model):
    plts,lobess=lung_plot(path='../Datasets/Rat PA Study/csv/5757.CF.PA63.pp2.specificVentilation.csv',model=model) 
    return (plts,)


@app.cell
def _(plts):
    plts[0],plts[1],plts[2],plts[3],plts[4]
    return


@app.cell
def _():
    def _():
        data=pd.read_csv('../Datasets/Rat PA Study/csv/5757.CF.PA63.pp2.specificVentilation.csv')
        return make_graphs_plots("",data)


    _()
    return


@app.cell
def _(model):
    plts2,_=lung_plot('../Datasets/Rat PA Study/csv/5767.WT.PA63.pp2.specificVentilation.csv',model)
    return (plts2,)


@app.cell
def _(plts2):
    plts2[0],plts2[1],plts2[2],plts2[3],plts2[4]
    return


@app.cell
def _(model):
    plts3,_=lung_plot('../Datasets/Rat Sterile Bead Study/csv/post_beads/4055.Phe508.beads.specificVentilation.csv',model)
    return (plts3,)


@app.cell
def _(plts3):
    plts3[0],plts3[1],plts3[2],plts3[3],plts3[4]
    return


if __name__ == "__main__":
    app.run()
