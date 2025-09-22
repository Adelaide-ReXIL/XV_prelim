

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import pandas as pd
    import numpy as np
    from scipy.spatial import Delaunay
    import networkx as nx
    import itertools
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize


@app.function
def plot_3d_points(df):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['x'], df['y'], df['z'], c=df['SV'], cmap='viridis', s=10)
    return plt.gca()


@app.function
def make_graph(data):
    data = data.sort_values(by=['x','y','z']).reset_index(drop=True)
    points = data[['x', 'y', 'z']].values
    tri = Delaunay(points)
    edges = set()
    for simplex in tri.simplices:
        for i, j in itertools.combinations(simplex, 2):
            edges.add((i, j))
    G = nx.Graph()
    for i, row in data.iterrows():
        G.add_node(i, SV=row['SV'],x_i=row['x'],y_i=row['y'], z=row['z'])
    G.add_edges_from(edges)
    return G


@app.function
def norm(data):
    for x in ['x', 'y', 'z']:
        data[x] = (data[x] - data[x].min()) / (data[x].max() - data[x].min())
    return data


@app.function
def plot_graph(data, G, edge=True):
    fixed_pos = {n: (G.nodes[n]['x_i'], G.nodes[n]['z']) for n in G.nodes}
    to_fix = {x: fixed_pos[x] for x in fixed_pos if np.random.rand() < 0.10}
    pos = nx.spring_layout(G,pos=fixed_pos,iterations=200,fixed=to_fix,k=1.1/np.sqrt(len(fixed_pos)))

    fig, ax = plt.subplots(figsize=(10, 10))

    if edge:
        for u, v in G.edges:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            ax.plot([x0, x1], [y0, y1], color='gray', linewidth=0.5, zorder=1)

    nodes = list(G.nodes)
    xs = [pos[n][0] for n in nodes]
    ys = [pos[n][1] for n in nodes]
    colors = [G.nodes[n]['SV'] for n in nodes]

    sc = ax.scatter(xs, ys, c=colors, s=10, cmap='viridis', zorder=2)
    fig.colorbar(sc, ax=ax, label='SV')

    df = pd.DataFrame([
        {'node': n, 'x': pos[n][0], 'y': pos[n][1], 'SV': G.nodes[n]['SV'], 's': G.nodes[n]['y_i']}
        for n in nodes
    ])
    return ax, df


@app.function
def make_graphs(G):
    fixed_pos = {n: (G.nodes[n]['x_i'], G.nodes[n]['z']) for n in G.nodes}
    to_fix = {x: fixed_pos[x] for x in fixed_pos if np.random.rand() < 0.25}
    pos = nx.spring_layout(G,pos=fixed_pos,iterations=250,fixed=to_fix,k=1.5/np.sqrt(len(fixed_pos)))


    nodes = list(G.nodes)

    df = pd.DataFrame([
        {'node': n, 'x': pos[n][0], 'y': pos[n][1], 'SV': G.nodes[n]['SV'], 's': G.nodes[n]['y_i']}
        for n in nodes
    ])
    return df


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():

    data=pd.read_csv('../Datasets/Synth/1.csv')
    data.columns=['SV','x','y','z']
    return (data,)


@app.cell
def _(data):
    plot_3d_points(data)
    return


@app.cell
def _(data):
    G=make_graph(data)
    return (G,)


@app.cell
def _(G):
    nx.to_numpy_array(G)
    return


@app.cell
def _(G, data):
    plot_graph(data,G,False)
    return


@app.cell
def _(G, data):
    plot_graph(data,G)
    return


@app.cell
def _():
    data2=pd.read_csv('../Datasets/Synth/2.csv')
    data2.columns=['SV','x','y','z']
    G2=make_graph(data=data2)
    plot_3d_points(data2)
    return G2, data2


@app.cell
def _(G2, data2):
    plot_graph(data=data2,G=G2,edge=False)
    return


@app.cell
def _():
    data3=pd.read_csv('../Datasets/Synth/3.csv')
    data3.columns=['SV','x','y','z']
    G3=make_graph(data=data3)
    plot_3d_points(data3)
    return G3, data3


@app.cell
def _(G3, data3):
    plot_graph(data=data3,G=G3,edge=False)
    return


@app.cell
def _():
    return


@app.function
def make_graphs_plots(filename,df=None,edge=False):
    data=None
    if df is not None:
        data=df
    else:
        data=pd.read_csv(f'../Datasets/Synth/{filename}')
    data.columns=['SV','x','y','z']
    norm(data)
    G=make_graph(data=data)
    d_plot=plot_3d_points(data)
    graph,df=plot_graph(data=data,G=G,edge=edge)
    return d_plot,graph,df


@app.function
def get_graphs(filename='',df=None):
    data=None
    if df is not None:
        data=df
    else:
        data=pd.read_csv(f'../Datasets/Synth/{filename}')
        data.columns=['SV','x','y','z']
    data.columns=['SV','x','y','z']
    norm(data)


    G=make_graph(data=data)
    df=make_graphs(G=G)
    return df


@app.cell
def _():
    make_graphs_plots('3.csv')
    return


@app.cell
def _():
    make_graphs_plots('4.csv')
    return


@app.cell
def _():
    make_graphs_plots('5.csv')
    return


@app.cell
def _():
    make_graphs_plots('6.csv')
    return


@app.cell
def _():
    make_graphs_plots('7.csv')
    return


@app.cell
def _():
    make_graphs_plots('8.csv')
    return


@app.cell
def _():
    make_graphs_plots('9.csv')
    return


@app.cell
def _():
    make_graphs_plots('10.csv')
    return


@app.cell
def _():
    make_graphs_plots('11.csv')
    return


@app.cell
def _():
    make_graphs_plots('12.csv')
    return


@app.cell
def _():
    make_graphs_plots('13.csv')
    return


@app.cell
def _():
    def _():
        data=pd.read_csv('../Datasets/Rat PA Study/csv/5757.CF.PA63.pp2.specificVentilation.csv')
        return make_graphs_plots("",data)

    _()
    return


if __name__ == "__main__":
    app.run()
