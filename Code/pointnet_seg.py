import marimo

__generated_with = "0.13.11"
app = marimo.App(width="full")

with app.setup:
    import torch
    from pointnetalt import PointNetPP,PointNetSetAbstraction
    import torch.nn as nn
    from torch.utils.data import Dataset,DataLoader
    import torch.nn.functional as F
    import torch.optim as optim
    from data_loader import XVData,csv_loader,PA_dataset_loader,bead_study_loader,sheep_dataset_loader,child_loader,adult_loader
    from Triangle_Test import plot_3d_points
    import pandas as pd
    import random
    import numpy as np
    from sklearn.metrics import f1_score, confusion_matrix, classification_report
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():

    def three_nn(xyz1, xyz2):
        """
        Input:
            xyz1: (B, N, 3)
            xyz2: (B, S, 3)
        Output:
            dists: (B, N, K)
            idx:   (B, N, K)
        """
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape
        dists = torch.cdist(xyz1, xyz2, p=2)  # (B, N, S)
        k = min(3, S)
        dists, idx = dists.topk(k, dim=-1, largest=False, sorted=False)
        return dists, idx

    def three_interpolate(points, idx, dists):
        """
        Interpolates features at `xyz1` from `xyz2`.
        Inputs:
            points: (B, C, S)
            idx:    (B, N, K)
            dists:  (B, N, K)
        Returns:
            interpolated: (B, C, N)
        """
        dists = torch.clamp(dists, min=1e-10)
        weight = 1.0 / dists
        weight = weight / weight.sum(dim=2, keepdim=True)

        B, C, S = points.shape
        B, N, K = idx.shape

        idx_exp = idx.unsqueeze(1).expand(-1, C, -1, -1)            # (B, C, N, K)
        points_exp = points.unsqueeze(2).expand(-1, -1, N, -1)      # (B, C, N, S)

        neighbor_feats = torch.gather(points_exp, 3, idx_exp)       # (B, C, N, K)
        interpolated = torch.sum(neighbor_feats * weight.unsqueeze(1), dim=3)  # (B, C, N)
        return interpolated

    class PointNetFeaturePropagation(nn.Module):
        def __init__(self, in_channel, mlp):
            super().__init__()
            layers = []
            last_channel = in_channel
            for out_channel in mlp:
                layers.append(nn.Conv1d(last_channel, out_channel, 1))
                layers.append(nn.BatchNorm1d(out_channel))
                layers.append(nn.ReLU())
                last_channel = out_channel
            self.mlp = nn.Sequential(*layers)

        def forward(self, xyz1, xyz2, feats1, feats2):
            """
            xyz1:   (B, N, 3)
            xyz2:   (B, S, 3)
            feats1: (B, C1, N)
            feats2: (B, C2, S)
            """
            if xyz2.shape[1] == 1 and feats2.shape[2] == 1:
                # Broadcast global feature if only one source point exists
                interpolated_feats = feats2.expand(-1, -1, xyz1.shape[1])  # (B, C2, N)
            else:
                dists, idx = three_nn(xyz1, xyz2)                          # (B, N, 3)
                interpolated_feats = three_interpolate(feats2, idx, dists)  # (B, C2, N)

            if feats1 is not None:
                feats = torch.cat([interpolated_feats, feats1], dim=1)     # (B, C1+C2, N)
            else:
                feats = interpolated_feats

            return self.mlp(feats)                                     # (B, mlp[-1], N)

    return PointNetFeaturePropagation, three_interpolate, three_nn


@app.cell
def _(PointNetFeaturePropagation, three_interpolate, three_nn):
    class PointNetSeg(nn.Module):
        def __init__(self, num_classes, pretrained_cls=None,contrast=False):
            super().__init__()
            self.contrast=contrast
            self.sa1 = pretrained_cls.sa1 if pretrained_cls else PointNetSetAbstraction(
                npoint=512, mlp=[8, 64, 64, 128], s_scale_factor=1, r_xyz=1.5,
                max_samples=128, use_geometry_centers=True, num_segments=8
            )
            self.sa2 = pretrained_cls.sa2 if pretrained_cls else PointNetSetAbstraction(
                npoint=128, mlp=[132, 128, 128, 256], s_scale_factor=1, r_xyz=0.75,
                max_samples=64, use_geometry_centers=True, num_segments=6
            )
            self.sa3 = pretrained_cls.sa3 if pretrained_cls else PointNetSetAbstraction(
                npoint=None, mlp=[256, 256, 512, 1024], s_scale_factor=1,
                r_xyz=0.5, max_samples=16, use_geometry_centers=False
            )

            self.fp2 = PointNetFeaturePropagation(in_channel=1024 + 256, mlp=[256, 256])
            self.fp1 = PointNetFeaturePropagation(in_channel=256 + 128, mlp=[256, 128])
            self.fp0 = PointNetFeaturePropagation(in_channel=128 + 8, mlp=[128, 128, 64])  # +8 for s_embed

            self.per_point_mask = nn.Sequential(
                nn.Conv1d(64, 32, 1),
                nn.ReLU(),
                nn.Conv1d(32, 1, 1),
                nn.Sigmoid()
            )

            self.input_embed = nn.Sequential(
                nn.Linear(4, 8),
                nn.ReLU(),
                nn.Linear(8, 4)
            )

            self.s_embed = nn.Sequential(
                nn.Linear(1, 8),
                nn.ReLU()
            )

        def forward(self, xyz):
            B, N, _ = xyz.shape
            l0_xyz = xyz
            l0_feats = xyz

            l1_xyz, l1_feats = self.sa1(l0_xyz, l0_feats)
            l2_xyz, l2_feats = self.sa2(l1_xyz, l1_feats)
            l3_xyz, l3_feats = self.sa3(l2_xyz, l2_feats)

            l1_feats = l1_feats.permute(0, 2, 1)
            l2_feats = l2_feats.permute(0, 2, 1)
            l3_feats = l3_feats.permute(0, 2, 1)

            l2_feats_up = self.fp2(l2_xyz, l3_xyz, l2_feats, l3_feats)
            l1_feats_up = self.fp1(l1_xyz, l2_xyz, l1_feats, l2_feats_up)

            dists, idx = three_nn(l1_xyz, l0_xyz)       # [B, 512, 3], [B, 512, 3]
            s = l0_xyz[..., :1]                         # [B, N, 1]
            s_nearest = three_interpolate(s.permute(0,2,1), idx, dists).permute(0,2,1)  # [B, 512, 1]
            s_feat = self.s_embed(s_nearest).permute(0, 2, 1)  # [B, 8, 512]

            l0_feats_up = self.fp0(l0_xyz, l1_xyz, None, torch.cat([l1_feats_up, 2*s_feat], dim=1))
            if self.contrast:
                return l0_feats_up

            point_mask = self.per_point_mask(l0_feats_up).squeeze(1)
            return point_mask

    return (PointNetSeg,)


@app.cell
def _():
    control, cf = child_loader()
    control.extend(adult_loader())

    random.shuffle(control)
    random.shuffle(cf)

    # test_cf,test_cntrl=cf.pop(),control.pop()

    # Split control
    n_control = len(control)
    n_control_train = int(0.8 * n_control)
    control_train = control[:n_control_train]
    control_test = control[n_control_train:]

    # Split cf
    n_cf = len(cf)
    n_cf_train = int(0.8 * n_cf)
    cf_train = cf[:n_cf_train]
    cf_test = cf[n_cf_train:]

    train_files = control_train + cf_train
    test_files = control_test + cf_test

    random.shuffle(train_files)
    random.shuffle(test_files)

    label_map = {f: 0 for f in control}
    label_map.update({f: 1 for f in cf})

    print(len(train_files), len(test_files))  # Total per class

    return label_map, test_files, train_files


@app.cell
def _():


    return


@app.cell
def _(PointNetSeg):
    model_pre:PointNetPP = torch.load("trained_8.pth",weights_only=False)

    model=PointNetSeg(2,model_pre)

    for block in [model.sa1, model.sa2, model.sa3]:
        for param in block.parameters():
            param.requires_grad = False


    def count_parameters(model):
        params = [p.numel() for p in model.parameters() if p.requires_grad]
        print(f'__\nParameters:{sum(params):>8}')
    count_parameters(model_pre)
    count_parameters(model)


    return model, model_pre


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.function
def weak_segmentation_loss(mask, labels):
    pred = mask.mean(dim=1)  # (B,)
    return F.binary_cross_entropy(pred, labels.float())


@app.function
def supervised_contrastive_loss(embeddings, labels, temperature=0.1):
    """
    embeddings: (B, D, N)
    labels:     (B,)
    """
    B, D, N = embeddings.shape
    features = embeddings.mean(dim=2)
    # features = features - features.mean(dim=0, keepdim=True)
    features = F.normalize(features, dim=1)
    # print(torch.std(features, dim=0)) 

    labels = labels.view(-1, 1)
    mask = labels.eq(labels.T).float()

    sim = torch.matmul(features, features.T) / temperature
    sim_exp = torch.exp(sim) - torch.eye(B, device=features.device)

    pos = sim_exp * mask
    denom = sim_exp.sum(dim=1, keepdim=True) + 1e-8

    loss = -torch.log((pos.sum(dim=1) + 1e-8) / denom.squeeze()).mean()
    var = torch.std(features, dim=0)
    var_loss = torch.mean((var - 1)**2)
    total_loss = loss + 0.1 * var_loss
    return loss


@app.cell
def _(label_map, model, test_files, train_files):

    device = torch.device('cuda:0')
    model.to(device)

    dataset = XVData(train_files+test_files, n=7500, map=label_map,frames=True)


    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=3, min_lr=1e-8)

    train_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=28, pin_memory=True, drop_last=True)

    num_epochs = 100
    patience = 10
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for points, labels in train_loader:
            points = points.to(device, non_blocking=True).float()    
            labels = labels.to(device, non_blocking=True).long()      

            optimizer.zero_grad()
            # print("Labels in batch:", labels.tolist())
            embeddings = model(points)   #(B,128,N)

            # loss = weak_segmentation_loss(embeddings, labels)
            loss=weak_segmentation_loss(embeddings,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()



        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     counter = 0
        #     torch.save(model.state_dict(), "best_seg_model.pt")
        # else:
        #     counter += 1
        #     if counter >= patience:
        #         print("Early stopping.")
        #         break

        print(f"Epoch {epoch+1:02d} | Train Loss: {running_loss / len(train_loader):.4f}")

    return (device,)


@app.cell
def _():
    return


@app.function
def visualise_model_output(model, data_files, label_map={}, k=3, n_points=7500):
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from torch.utils.data import DataLoader
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 

    device=torch.device('cuda:0')

    test_dataset = XVData(data_files, n=n_points, map=label_map, frames=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    model.eval()
    all_coords, all_feats, all_sv = [], [], []

    with torch.no_grad():
        for points, _ in test_loader:
            points = points.to(device)
            feats = model(points)[0].T.cpu().numpy()     # (N, D)
            coords = points[0, :, 1:].cpu().numpy()      # (N, 3)
            sv = points[0, :, 0].cpu().numpy()           # (N,)
            all_feats.append(feats)
            all_coords.append(coords)
            all_sv.append(sv)
            break  # one sample only

    features = np.concatenate(all_feats, axis=0)  # (N, D)
    coords = np.concatenate(all_coords, axis=0)   # (N, 3)
    sv = np.concatenate(all_sv, axis=0)           # (N,)

    # PCA projection + normalization
    pca = PCA(n_components=1)
    pca_colors = pca.fit_transform(features).squeeze()
    pca_colors = (pca_colors - pca_colors.min()) / (pca_colors.max() - pca_colors.min() + 1e-8)

    # K-means clustering
    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
    cluster_labels = kmeans.fit_predict(features)

    # Plotting
    def plot_triple_3d(coords, sv, pca_vals, clusters):
        fig = plt.figure(figsize=(18, 5))

        ax1 = fig.add_subplot(131, projection='3d')
        sc1 = ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=sv, cmap='RdYlGn', s=1)
        ax1.set_title("SV (Input Feature)")
        plt.colorbar(sc1, ax=ax1, fraction=0.046)

        ax2 = fig.add_subplot(132, projection='3d')
        sc2 = ax2.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=-pca_vals, cmap='RdYlGn', s=1)
        ax2.set_title("PCA on Learned Features")
        plt.colorbar(sc2, ax=ax2, fraction=0.046)

        ax3 = fig.add_subplot(133, projection='3d')
        sc3 = ax3.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=-clusters, cmap='RdYlGn', s=1)
        ax3.set_title(f"K-means Clustering (k={k})")
        plt.colorbar(sc3, ax=ax3, fraction=0.046)

        plt.tight_layout()
        return plt.gca()

    return plot_triple_3d(coords, sv, pca_colors, cluster_labels)


@app.cell
def _():
    return


@app.cell
def _():
    child_loader(),adult_loader()
    return


@app.cell
def _():
    from augumented_lung import aug_lung

    return (aug_lung,)


@app.cell
def _(aug_lung, cmap):



    def visualise_point_mask(points, mask, title="Segmentation Output"):
        points = points.detach().cpu()     # (N, 4)
        mask = mask.detach().cpu()         # (N,)

        # mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)  # normalize to [0,1]
        mask=mask.squeeze()
        df = pd.DataFrame(torch.cat([mask.unsqueeze(1), points[:, 1:]], dim=1).numpy(), columns=["SV", "x", "y", "z"])
        df=aug_lung(df, 5)
        features = df[["SV", "x", "y", "z"]].values
        kmeans = KMeans(n_clusters=4).fit(features)
        df["cluster"] = kmeans.labels_

        fig = plt.figure(figsize=(12, 5))

        # Plot mask
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        sc1 = ax1.scatter(df["x"], df["y"], df["z"], c=df["SV"], cmap=cmap, s=1,vmin=0,vmax=1)
        ax1.set_title("Mask Intensity")
        fig.colorbar(sc1, ax=ax1, shrink=0.5)
        ax1.set_facecolor('white')
        ax1.set_axis_off()

        # Plot clusters
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        sc2 = ax2.scatter(df["x"], df["y"], df["z"], c=df["cluster"], cmap=cmap, s=1)
        ax2.set_title("KMeans Clusters")
        fig.colorbar(sc2, ax=ax2, shrink=0.5)
        ax2.set_axis_off()
        ax2.set_facecolor('white')

        plt.tight_layout()

        return plt.gca(),df
    return (visualise_point_mask,)


@app.cell
def _():
    # visualise_model_output(model,["../Datasets/XV Clinical Data/adult_controls_from_Miami/FLASH10010/FLASH10010-SUPINE-INSP/FLASH10010-INSP_final.csv"])
    return


@app.cell
def _():
    return


@app.function
def get_saliency_from_classifier(model, sample, label_idx):
    model.eval()
    sample = sample.unsqueeze(0).requires_grad_(True)  # (1, N, 4)

    out = model(sample)  # (1, 2)
    score = out[0, label_idx]
    score.backward()

    saliency = sample.grad.norm(dim=2).squeeze(0)  # (N,)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency.cpu(), sample.detach().squeeze(0).cpu()


@app.function
def plot_saliency(pts, saliency):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(pts[:,1], pts[:,2], pts[:,3], c=saliency.detach(), cmap='RdYlGn', s=1)
    fig.colorbar(p)
    plt.show()


@app.cell
def _():
    from torch.autograd import grad

    def integrated_gradients(model, sample, label_idx, steps=20):
        model.eval()
        sample = sample.unsqueeze(0).requires_grad_(True)  # (1, N, 4)
        baseline = torch.zeros_like(sample)

        total_grad = torch.zeros_like(sample)
        for alpha in torch.linspace(0, 1, steps):
            interpolated = baseline + alpha * (sample - baseline)
            interpolated.requires_grad_(True)
            out = model(interpolated)
            score = out[0, label_idx]
            grad_ = grad(score, interpolated, retain_graph=True)[0]
            total_grad += grad_

        avg_grad = total_grad / steps
        saliency = ((sample - baseline) * avg_grad).norm(dim=2).squeeze(0)
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        return saliency.cpu(), sample.squeeze(0).detach().cpu()

    return


@app.cell
def _(model_pre):
    def forward_with_feats(self, xyz):
        l0_xyz = xyz
        l0_feats = xyz

        l1_xyz, l1_feats = self.sa1(l0_xyz, l0_feats)
        l2_xyz, l2_feats = self.sa2(l1_xyz, l1_feats)
        l3_xyz, l3_feats = self.sa3(l2_xyz, l2_feats)

        global_feat = l3_feats.max(dim=1)[0]
        logits = self.classifier(global_feat)

        return logits, [l1_xyz, l1_feats, l2_xyz, l2_feats, l3_xyz, l3_feats]

    import types
    model_pre.forward_with_feats = types.MethodType(forward_with_feats, model_pre)

    return


@app.function
def input_times_grad(model, sample, label_idx):
    model.eval()
    sample = sample.unsqueeze(0).requires_grad_(True)
    out = model(sample)
    score = out[0, label_idx]
    score.backward()
    saliency = (sample.grad * sample).norm(dim=2).squeeze(0)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency.cpu(), sample.detach().squeeze(0).cpu()


@app.function
def get_attention_saliency(model, sample):
    model.eval()
    with torch.no_grad():
        _, feats = model.forward_with_feats(sample.unsqueeze(0).to(next(model.parameters()).device))
        l1_xyz, l1_feats, _, _ = feats
        saliency = l1_feats.norm(dim=2).squeeze(0)
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        return saliency.cpu(), l1_xyz.squeeze(0).cpu()


@app.cell
def _():
    test=XVData(["../Datasets/XV Clinical Data/WCH-CF-10005-20230414/WCH-CF-10005-SUPINE-INSP/WCH-CF-10005-INSP_final.csv"],n=10000,frames=True)
    p, l = next(iter(test))

    # saliency, pts =get_attention_saliency(model_pre, p.to(device))
    lung = pd.DataFrame(test[0][0].numpy(), columns=["SV", "x", "y", "z"])
    plot_3d_points(lung)
    return (p,)


@app.function
def get_all_saliency_dfs(model, sample, aug_fn, aug_val=10):
    model.eval()
    with torch.no_grad():
        _, feats = model.forward_with_feats(sample.unsqueeze(0).to(next(model.parameters()).device))
        l1_xyz, l1_feats, l2_xyz, l2_feats, l3_xyz, l3_feats = feats

        def process_layer(feats, xyz):
            if xyz.shape[1] <= 1:
                return None  # skip global-only layer
            sal = feats.norm(dim=2).squeeze(0)
            # sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
            df = pd.DataFrame(torch.cat([sal.unsqueeze(1), xyz.squeeze(0)[:, 1:]], dim=1).cpu().numpy(),
                              columns=["SV", "x", "y", "z"])
            return aug_fn(df, aug_val)

        df1 = process_layer(l1_feats, l1_xyz)
        df2 = process_layer(l2_feats, l2_xyz)
        df3 = process_layer(l3_feats, l3_xyz)

        return df1, df2, df3


@app.cell
def _(aug_lung, device, model_pre, p):
    a,b,_=get_all_saliency_dfs(model_pre,p.to(device),aug_lung)
    a = a[~((a[['x', 'y', 'z']] == 0).all(axis=1))]
    b = b[~((b[['x', 'y', 'z']] == 0).all(axis=1))]
    a['SV'] = (a['SV'] - a['SV'].min()) / (a['SV'].max() - a['SV'].min())
    b['SV'] = (b['SV'] - b['SV'].min()) / (b['SV'].max() - b['SV'].min())

    return a, b


@app.cell
def _(a, b):
    plot_3d_points(a),plot_3d_points(b)
    return


@app.cell
def _(device, model, p, visualise_point_mask):
    visualise_point_mask(p,model(p.to(device).unsqueeze(0)))
    return


@app.cell
def _(aug_lung, pts, saliency):
    df = pd.DataFrame(torch.cat([saliency.unsqueeze(1), pts[:, 1:]], dim=1).numpy(), columns=["SV", "x", "y", "z"])
    df=aug_lung(df,10)

    return


@app.cell
def _(pts, saliency):
    plot_saliency(pts,saliency)
    return


@app.cell
def _():
    from Lung_segment_Test import norm
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("light_to_dark_red", ["#fde0dd", "#fa9fb5", "#c51b8a", "#7a0177"])



    return (cmap,)


@app.cell
def _(a, b, cmap):


    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(a["x"], a["y"], a["z"], c=a["SV"], cmap=cmap, s=10)
    ax.set_axis_off()
    fig.colorbar(sc, ax=ax, label="Attention")
    plt.tight_layout()
    ax.set_facecolor('white')
    plt.show()


    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(b["x"], b["y"], b["z"], c=b["SV"], cmap=cmap, s=10)
    fig.colorbar(sc, ax=ax, label="Attention")
    ax.set_axis_off()
    ax.set_facecolor('white')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    # torch.save(model, 'seg_model3.pth')
    return


@app.cell
def _(device, p):
    def _():
        from fvcore.nn import FlopCountAnalysis

        model = torch.load("trained_8.pth", weights_only=False)
        model.eval()

        dummy_input = p.to(device).unsqueeze(0)  # Adjust shape to your input, e.g. (B, N, 3)
        flops = FlopCountAnalysis(model, dummy_input)
        print(f"{flops.total() / 1e9:.2f} GFLOPs")
    _()
    return


if __name__ == "__main__":
    app.run()
