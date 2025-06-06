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

    return (PointNetFeaturePropagation,)


@app.cell
def _(PointNetFeaturePropagation):

    class PointNetSeg(nn.Module):
        def __init__(self, num_classes, pretrained_cls=None):
            super().__init__()

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

            # Decoder (Feature Propagation)
            self.fp2 = PointNetFeaturePropagation(in_channel=1024 + 256, mlp=[256, 256])
            self.fp1 = PointNetFeaturePropagation(in_channel=256 + 128, mlp=[256, 128])
            self.fp0 = PointNetFeaturePropagation(in_channel=128 ,   mlp=[128, 128, 128])

            self.projector = nn.Sequential(
                nn.Conv1d(128, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, 64, 1)  # final embedding dimension
            )


            # Per-point classifier
            self.conv1 = nn.Conv1d(128, 128, 1)
            self.bn1   = nn.BatchNorm1d(128)
            self.drop1 = nn.Dropout(0.5)
            self.conv2 = nn.Conv1d(128, num_classes, 1)

        def forward(self, xyz):
            # xyz: (B, N, 4)
            B, N, _ = xyz.shape

            l0_xyz = xyz
            l0_feats = xyz

            # Encoder
            l1_xyz, l1_feats = self.sa1(l0_xyz, l0_feats)           # (B, 512, 4), (B, 512, 128)        
            l2_xyz, l2_feats = self.sa2(l1_xyz, l1_feats)           # (B, 128, 4), (B, 128, 256)
            l3_xyz, l3_feats = self.sa3(l2_xyz, l2_feats)           # (B, 1, 1024)

            # Transpose features to (B, C, N)
            l1_feats = l1_feats.permute(0, 2, 1)    # (B, 128, 512)
            l2_feats = l2_feats.permute(0, 2, 1)    # (B, 256, 128)
            l3_feats = l3_feats.permute(0, 2, 1)    # (B, 1024, 1)

            # Decoder (Feature Propagation)
            l2_feats_up = self.fp2(l2_xyz, l3_xyz, l2_feats, l3_feats)    # (B, 256, 128)
            l1_feats_up = self.fp1(l1_xyz, l2_xyz, l1_feats, l2_feats_up) # (B, 128, 512)
            l0_feats_up = self.fp0(l0_xyz, l1_xyz, None, l1_feats_up)     # (B, 128, N)

            return l0_feats_up

    return (PointNetSeg,)


@app.cell
def _():
    control, cf = child_loader()
    control.extend(adult_loader())

    random.shuffle(control)
    random.shuffle(cf)

    min_train = int(0.8 * min(len(control), len(cf)))
    min_test = int(0.2 * min(len(control), len(cf)))

    control_train = control[:min_train]
    cf_train = cf[:min_train]
    control_test = control[min_train:min_train + min_test]
    cf_test = cf[min_train:min_train + min_test]

    train_files = control_train + cf_train
    test_files = control_test + cf_test

    random.shuffle(train_files)

    random.shuffle(test_files)

    label_map = {}

    for f in control_train + control_test:
        label_map[f] = 0
    for f in cf_train + cf_test:
        label_map[f] = 1
    print(len(control_train) + len(control_test), len(cf_train) + len(cf_test))
    return label_map, train_files


@app.cell
def _(PointNetSeg):
    model_pre:PointNetPP = torch.load("trained_5.pth",weights_only=False)

    model=PointNetSeg(2,model_pre)

    for block in [model.sa1, model.sa2, model.sa3]:
        for param in block.parameters():
            param.requires_grad = False



    return model, model_pre


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
def _(label_map, model, train_files):

    device = torch.device('cuda:0')
    model.to(device)

    dataset = XVData(train_files, n=7500, map=label_map,frames=True)


    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=3, min_lr=1e-8)

    train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=28, pin_memory=True, drop_last=True)

    num_epochs = 50
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

            loss = supervised_contrastive_loss(embeddings, labels)
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
    child_loader(),adult_loader()
    return


@app.cell
def _(model):
    visualise_model_output(model,["../Datasets/XV Clinical Data/WCH-CF-10011-20231013/WCH-CF-10011-SUPINE-INSP/WCH-CF-10011-INSP_final.csv"])
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

        return logits, [l1_xyz, l1_feats, l2_xyz, l2_feats]

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
def _(device, model_pre):
    test=XVData(["../Datasets/XV Clinical Data/WCH-CF-10010-20231006/WCH-CF-10010-SUPINE-INSP/WCH-CF-10010-INSP_final.csv"],n=7500,frames=True)
    p, l = next(iter(test))

    saliency, pts =get_attention_saliency(model_pre, p.to(device))

    return pts, saliency


@app.cell
def _():
    from augumented_lung import aug_lung

    return (aug_lung,)


@app.cell
def _(pts, saliency):
    df = pd.DataFrame(torch.cat([-saliency.unsqueeze(1), pts[:, 1:]], dim=1).numpy(), columns=["SV", "x", "y", "z"])

    return (df,)


@app.cell
def _(pts, saliency):
    plot_saliency(pts,saliency)
    return


@app.cell
def _(aug_lung, df):
    plot_3d_points(aug_lung(df,10))
    return


if __name__ == "__main__":
    app.run()
