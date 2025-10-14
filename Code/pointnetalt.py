import marimo

__generated_with = "0.13.11"
app = marimo.App(width="columns")

with app.setup:
    # Initialization code that runs before all other cells
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    # import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset,DataLoader,random_split
    import trimesh
    import gc
    from Triangle_Test import plot_3d_points
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import random
    import pickle
    with open('model_normal.pkl', 'rb') as f:
        model_diff = pickle.load(f)


@app.cell
def _():
    import marimo as mo

    return


@app.class_definition
class PointCloud(Dataset):
    def __init__(self, root_dir, n=1024, train=True):
        self.classes = os.listdir(root_dir)
        self.root_dir = root_dir
        self.n = n
        self.labels = {cls: i for i, cls in enumerate(self.classes)}
        self.train = train

        trainls = []
        for cls in self.classes:
            folderpath = os.path.join(self.root_dir, cls, "train")
            trainls.extend([os.path.join(folderpath, f) for f in os.listdir(folderpath)])
        self.trainfilepath = pd.DataFrame(trainls)

        testls = []
        for cls in self.classes:
            folderpath = os.path.join(self.root_dir, cls, "test")
            testls.extend([os.path.join(folderpath, f) for f in os.listdir(folderpath)])
        self.testfilepath = pd.DataFrame(testls)

    def __len__(self):
        return len(self.trainfilepath) if self.train else len(self.testfilepath)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path = self.trainfilepath.iloc[idx][0] if self.train else self.testfilepath.iloc[idx][0]
        mesh = trimesh.load(path)
        pts = mesh.sample(self.n)  # (n, 3)


        # normalize x, y, z to 0..1
        mins = pts.min(axis=0, keepdims=True)
        maxs = pts.max(axis=0, keepdims=True)
        norm_pts = (pts - mins) / (maxs - mins + 1e-8)

        s = model_diff.predict(pd.DataFrame(norm_pts, columns=[ "x", "y", "z"]))
        s = (s - s.min()) / (s.max() - s.min() + 1e-8)
        pts_out = np.hstack([s.reshape(-1, 1), norm_pts])  # (n, 4)

        label = np.zeros(len(self.classes))
        for cls in self.labels:
            if cls in path:
                label[self.labels[cls]] = 1

        return torch.tensor(pts_out, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


@app.cell
def _():
    train_data=PointCloud('../Datasets/archive/ModelNet10')
    return (train_data,)


@app.cell
def _():
    return


@app.cell
def _(train_data):
    sample=train_data[11][0]
    df = pd.DataFrame(sample.numpy(), columns=["SV", "x", "y", "z"])

    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(df):
    plot_3d_points(df)
    return


@app.function
def ball_query_4d_dynamic_s(coords4d, centers4d, r_xyz, max_samples,s_thresh=1.25):
    """
    4D ball‐query where r_s (the radius along the 's' axis) is computed per-center:
      r_s^(k) = 1.5 * s_center_k

    We keep neighbors j of center k if
      | s_j – s_center_k | < r_s^(k)
    AND
      √[ (x_j – x_ck)^2 + (y_j – y_ck)^2 + (z_j – z_ck)^2 ] < r_xyz

    Inputs:
      coords4d:   (B, N, 4)  floats, coords4d[...,0]=s, coords4d[...,1:4]=(x,y,z)
      centers4d:  (B, S, 4)  floats, similarly (s_center, x_center, y_center, z_center)
      r_xyz:      float      fixed spatial radius for (x,y,z) neighborhood
      max_samples: int       maximum #neighbors per center
    Returns:
      neighbor_idx: (B, S, max_samples)  long indices of neighbors in [0..N)
      valid_mask:   (B, S, max_samples)  0/1 mask (1 = within both radii)
    """
    B, N, _ = coords4d.shape
    S = centers4d.shape[1]

    # 1) Extract s_j and (x,y,z)_j for all points
    s_all   = coords4d[..., 0].unsqueeze(1)   # (B, 1, N)
    xyz_all = coords4d[..., 1:]               # (B, N, 3)

    s_centers   = centers4d[..., 0]           # (B, S)
    xyz_centers = centers4d[..., 1:]          # (B, S, 3)

    # 2) Compute per-center s‐radius: r_s_dyn[b,k] = 1.5 * s_centers[b,k]
    r_s_dyn = torch.clamp(s_thresh * s_centers, min=0.0001, max=0.5)           # (B, S)

    # 3) Compute |s_j – s_center_k| for all j,k:
    #    shape (B, S, N): 
    diff_s = s_centers.unsqueeze(-1) - s_all    # (B, S, N)
    abs_diff_s = torch.abs(diff_s)              # (B, S, N)

    # 4) Compute spatial squared distance: ||(x,y,z)_j – (x,y,z)_ck||^2
    xyz_centers_exp = xyz_centers.unsqueeze(2)  # (B, S, 1, 3)
    xyz_all_exp     = xyz_all.unsqueeze(1)       # (B, 1, N, 3)
    diff_xyz = xyz_centers_exp - xyz_all_exp     # (B, S, N, 3)
    dxyz2    = torch.sum(diff_xyz * diff_xyz, dim=-1)  # (B, S, N)

    # 5) Build boolean masks for each criterion
    #    mask_s[b,k,j]  = True if |s_j – s_ck| < r_s_dyn[b,k]
    #    mask_xyz[b,k,j]= True if dxyz2[b,k,j] < (r_xyz)^2
    mask_s   = abs_diff_s <r_s_dyn.unsqueeze(-1)        # (B, S, N)
    mask_xyz = dxyz2 < (r_xyz * r_xyz)                    # (B, S, N)

    # 6) Combined mask: valid[b,k,j] = mask_s AND mask_xyz
    valid = mask_s & mask_xyz                             # (B, S, N), bool

    # 7) To select up to max_samples neighbors, set invalid distances to a large value
    INF = 1e10
    # We only care about ordering by whether valid or not; use INF to push invalids to the back
    weighted_dist2 = torch.zeros_like(dxyz2)  # (B, S, N)
    # For any valid (b,k,j), we keep the actual squared spatial distance dxyz2; else INF
    weighted_dist2[valid] = dxyz2[valid]
    weighted_dist2[~valid] = INF

    # 8) Take the top‐k smallest “distances” along N
    #    idx[b,k,:] are indices of the nearest max_samples neighbors
    _, idx = weighted_dist2.topk(
        k=max_samples,
        dim=-1,
        largest=False,
        sorted=False
    )  # idx: (B, S, max_samples)

    # 9) Rebuild valid_mask: check again whether each chosen idx[b,k,m] was truly valid
    gathered_valid = valid.gather(dim=-1, index=idx)     # (B, S, max_samples)
    valid_mask = gathered_valid.long()                    # convert to 0/1

    # valid_counts = valid_mask[0].sum(dim=-1)  # (S,)
    # print("Min / Mean / Max neighbors per center:", valid_counts.min().item(), valid_counts.float().mean().item(), valid_counts.max().item())

    return idx, valid_mask


@app.function
def geometry_centers(coords, num_segments=6):
    B, N, _ = coords.shape
    device = coords.device
    bins = torch.linspace(0, 1, num_segments + 1, device=device)

    mins = coords.amin(dim=1, keepdim=True)
    maxs = coords.amax(dim=1, keepdim=True)
    ranges = (maxs - mins).clamp(min=1e-6)
    normed = (coords - mins) / ranges

    idx = torch.bucketize(normed, bins) - 1
    idx = idx.clamp(min=0, max=num_segments - 1)

    flat_idx = idx[:, :, 0] * num_segments**2 + idx[:, :, 1] * num_segments + idx[:, :, 2]
    centers = torch.zeros(B, num_segments**3, 3, device=device)
    counts = torch.zeros(B, num_segments**3, 1, device=device)

    for b in range(B):
        centers[b].index_add_(0, flat_idx[b], coords[b])
        counts[b].index_add_(0, flat_idx[b], torch.ones_like(coords[b, :, :1]))

    centers = centers / counts.clamp(min=1)
    return centers.view(B, num_segments**3, 3)


@app.cell
def _():
    from mpl_toolkits.mplot3d import Axes3D


    def visualize_ball_query(coords, centers, neighbor_indices, valid_mask, sample_centers=10):
        coords = coords[0].cpu()
        centers = centers[0].cpu()
        neighbor_indices = neighbor_indices[0].cpu()
        valid_mask = valid_mask[0].cpu()

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=1, c='lightgray', alpha=0.5, label='Points')
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='red', s=10, label='Centers')

        # Plot a few center-neighbor sets
        for i in torch.randperm(centers.shape[0])[:sample_centers]:
            c = centers[i]
            valid = valid_mask[i]
            neighbors = coords[neighbor_indices[i][valid]]
            ax.scatter(neighbors[:, 0], neighbors[:, 1], neighbors[:, 2], s=4)

            for n in neighbors:
                ax.plot([c[0], n[0]], [c[1], n[1]], [c[2], n[2]], c='blue', linewidth=0.5, alpha=0.5)

        ax.legend()
        plt.show()
    def plot_3d_points_with_centers(df_points, df_centers):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df_points["x"], df_points["y"], df_points["z"], c='b', s=1, label='Points')
        ax.scatter(df_centers["x"], df_centers["y"], df_centers["z"], c='r', s=20, label='Centers')
        ax.legend()
        plt.show()




    # coords4d = sample.unsqueeze(0)  # [1, N, 4]

    # coords3d = coords4d[..., 1:] 
    # centers3d = geometry_centers(coords3d, 6).view(1, -1, 3)
    # S = centers3d.shape[1]

    # diff3d = centers3d.unsqueeze(2) - coords3d.unsqueeze(1)  
    # dist3d = torch.sum(diff3d * diff3d, dim=-1)            
    # _, idx_min = torch.min(dist3d, dim=-1)                
    # batch_idx = torch.arange(1, device=coords4d.device).view(1, 1).expand(1, S)
    # s_centers = coords4d[batch_idx, idx_min, 0]             

    # centers4d = torch.cat([
    #     s_centers.unsqueeze(-1),    # (B, S, 1)
    #     centers3d                   # (B, S, 3)
    # ], dim=-1)   
    # neighbor_indices, valid_mask = ball_query_4d_dynamic_s(coords4d, centers4d, r_xyz=1, max_samples=64)

    # visualize_ball_query(coords4d[..., 1:], centers4d[..., 1:], neighbor_indices, valid_mask, sample_centers=centers4d.shape[1])


    return


@app.class_definition
class PointNetSetAbstraction(nn.Module):
    def __init__(
        self,
        npoint,
        mlp,                  # mlp[0] must = 8 = (4 rel‐coords + 4 orig feats)
        s_scale_factor=1.5,   # multiply each center’s s by this to get its r_s
        r_xyz=0.5,            # fixed radius for spatial (x,y,z)
        max_samples=32,
        use_geometry_centers=False,
        num_segments=6,
        s_t=1.5
    ):
        super().__init__()
        self.npoint = npoint
        self.s_scale_factor = s_scale_factor
        self.r_xyz = r_xyz
        self.max_samples = max_samples
        self.use_geometry_centers = use_geometry_centers
        self.num_segments = num_segments
        self.s_t=s_t

        # Build the MLP: input channels = mlp[0] (e.g. 8) → ...
        layers = []
        in_channel = mlp[0]
        for out_channel in mlp[1:]:
            layers += [
                nn.Conv1d(in_channel, out_channel, 1),
                nn.BatchNorm1d(out_channel),
                nn.ReLU()
            ]
            in_channel = out_channel
        self.mlp = nn.Sequential(*layers)

    def forward(self, coords4d, features4d):
        """
        coords4d:   (B, N, 4)   # (s, x, y, z)
        features4d: (B, N, 4)   # same (s,x,y,z)
        """
        B, N, _ = coords4d.shape

        if self.npoint is not None:
            # 1) Extract purely 3D coords for geometry‐centers
            coords3d = coords4d[..., 1:]  # (B, N, 3)

            # 2) Pick S centers (in 3D) and assign each an s_center
            if self.use_geometry_centers:
                centers3d = geometry_centers(coords3d, self.num_segments).view(B, -1, 3)
                S = centers3d.shape[1]

                # Find nearest neighbor in 3D to copy its s
                diff3d = centers3d.unsqueeze(2) - coords3d.unsqueeze(1)  # (B, S, N, 3)
                dist3d = torch.sum(diff3d * diff3d, dim=-1)             # (B, S, N)
                _, idx_min = torch.min(dist3d, dim=-1)                  # (B, S)
                batch_idx = torch.arange(B, device=coords4d.device).view(B, 1).expand(B, S)
                s_centers = coords4d[batch_idx, idx_min, 0]             # (B, S)

                centers4d = torch.cat([
                    s_centers.unsqueeze(-1),    # (B, S, 1)
                    centers3d                   # (B, S, 3)
                ], dim=-1)                       # → (B, S, 4)
            else:
                idx = torch.randint(N, (B, self.npoint), device=coords4d.device)  # (B, S)
                batch_idx = torch.arange(B, device=coords4d.device).view(B, 1).expand(B, self.npoint)
                centers4d = coords4d[batch_idx, idx]  # (B, S, 4)
                S = self.npoint

            # 3) Run the dynamic‐s ball‐query
            neighbor_idx, valid_mask = ball_query_4d_dynamic_s(
                coords4d,      # (B, N, 4)
                centers4d,     # (B, S, 4)
                self.r_xyz,    # fixed xyz‐radius
                self.max_samples,
                s_thresh=self.s_t
            )  # neighbor_idx: (B, S, k), valid_mask: (B, S, k)

            # 4) Gather neighbor coords & features
            batch_idx2 = torch.arange(B, device=coords4d.device).view(B, 1, 1).expand(B, S, self.max_samples)
            neigh_coords4d = coords4d[batch_idx2, neighbor_idx]   # (B, S, k, 4)
            neigh_feats4d  = features4d[batch_idx2, neighbor_idx] # (B, S, k, 4)

            # 5) Compute relative 4D offsets: (Δs, Δx, Δy, Δz)
            rel4d = neigh_coords4d - centers4d.unsqueeze(2)       # (B, S, k, 4)

            # 6) Concatenate rel4d (4 dims) + neigh_feats4d (4 dims) → 8 channels
            combined = torch.cat((rel4d, neigh_feats4d), dim=-1)   # (B, S, k, 8)
            combined = combined * valid_mask.unsqueeze(-1).float()

            # 7) Reshape for Conv1d: (B*S, 8, k)
            x = combined.permute(0, 1, 3, 2).reshape(B * S, -1, self.max_samples)

            # 8) MLP + max‐pool over k
            x = self.mlp(x)                    # (B*S, C_out, k)
            x = torch.max(x, 2)[0].view(B, S, -1)  # (B, S, C_out)

            return centers4d, x

        # If npoint is None: global pooling over all N points
        x = self.mlp(features4d.permute(0, 2, 1))       # (B, C_out, N)
        x = torch.max(x, 2, keepdim=True)[0].permute(0, 2, 1)  # (B, 1, C_out)
        return coords4d[:, :1], x


@app.class_definition
class PointNetPP(nn.Module):
    def __init__(self, input_dim=4, num_classes=2):
        super().__init__()

        # SA1: mlp[0]=8, use dynamic s‐radius = 1.5 * s_center, r_xyz=0.5
        self.sa1 = PointNetSetAbstraction(
            npoint=1024,
            mlp=[8, 64, 64, 128],
            s_scale_factor=1,
            r_xyz=1.25,
            max_samples=128,
            use_geometry_centers=True,
            num_segments=8,
            s_t=1.25
        )

        # SA2: input = (l1_feats 128) + 4 dims = 132, dynamic s‐radius=1.5*s_center, r_xyz=0.25
        self.sa2 = PointNetSetAbstraction(
            npoint=512,
            mlp=[132, 128, 128, 256],
            s_scale_factor=1,
            r_xyz=0.75,
            max_samples=64,
            use_geometry_centers=True,
            num_segments=6,
            s_t=1.5
        )

        # SA3: global, no query needed
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            mlp=[256, 256, 512, 1024],
            s_scale_factor=1,  # not used because npoint=None
            r_xyz=1.5,
            max_samples=16,
            use_geometry_centers=False,
            s_t=1.25
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        x: (B, N, 4)   # x[...,0]=s, x[...,1:4]=(x,y,z)
        """
        coords4d = x       # (s,x,y,z)
        feats4d  = x       # same 4D

        # SA1: centers from 3D, dynamic s-radius in 4D
        l1_centers4d, l1_feats = self.sa1(coords4d, feats4d)
        # l1_centers4d: (B,512,4), l1_feats: (B,512,128)

        # SA2: concat (4D center + 128D feat) = 132 channels

        l2_centers4d, l2_feats = self.sa2(l1_centers4d, l1_feats)
        # l2_centers4d: (B,128,4), l2_feats: (B,128,256)

        # SA3: global
        _, l3_feats = self.sa3(l2_centers4d, l2_feats)  # (B, 1, 1024)

        # Classifier on global feature
        return self.classifier(l3_feats.view(x.shape[0], -1))


@app.cell
def _():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return


@app.cell
def _():
    device = torch.device('cuda:0')

    dataset = PointCloud('../Datasets/archive/ModelNet10')
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    model = PointNetPP(input_dim=4, num_classes=len(dataset.classes)).to(device)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=28, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=28, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 50
    patience = 10
    best_val_loss = float('inf')
    counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for points, labels in train_loader:
            points = points.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).argmax(dim=1).long()

            optimizer.zero_grad()
            outputs = model(points)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for points, labels in val_loader:
                points = points.to(device, non_blocking=True).float()
                labels = labels.to(device, non_blocking=True).argmax(dim=1).long()
                outputs = model(points)
                val_loss += criterion(outputs, labels).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1} Train Loss: {running_loss / len(train_loader):.4f}  Val Loss: {val_loss:.4f}")

        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     counter = 0
        # else:
        #     counter += 1
        #     if counter >= patience:
        #         print("Early stopping triggered.")
        #         break
    return (model,)


@app.cell
def _(model):
    torch.save(model, 'pointnetpp_4d_full13.pth')
    return


@app.cell
def _(model):
    def _():
        from sklearn.metrics import f1_score, confusion_matrix, classification_report
        import time

        device = torch.device('cuda:0')
        # model = torch.load('pointnetpp_4d_full6.pth', weights_only=False)
        model.to(device)

        test = PointCloud('../Datasets/archive/ModelNet10', train=False)
        testloader = DataLoader(test, batch_size=64, shuffle=False, num_workers=16, pin_memory=True)

        model.eval()
        correct = total = 0

        with torch.no_grad():
            for points, labels in testloader:
                points = points.to(device, non_blocking=True).float()
                labels = labels.to(device, non_blocking=True).argmax(dim=1).long()
                outputs = model(points)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        print(f"Test Accuracy: {100 * correct / total:.2f}%")

        test = PointCloud('../Datasets/archive/ModelNet10', train=False)
        points, label = test[0]
        points = points.unsqueeze(0).to(device).float()
        label = label.argmax().item()

        start = time.time()
        with torch.no_grad():
            output = model(points)
        end = time.time()

        pred = output.argmax(dim=1).item()
        print(f"True label: {label}, Predicted: {pred}, Time: {(end - start) * 1000:.2f} ms")

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for points, labels in testloader:
                points = points.to(device, non_blocking=True).float()
                labels = labels.to(device, non_blocking=True).argmax(dim=1).long()
                outputs = model(points)
                preds = outputs.argmax(dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        acc = (all_preds == all_labels).mean() * 100
        print(f"Accuracy: {acc:.2f}%")
        print(f"F1 Score (macro): {f1_score(all_labels, all_preds, average='macro'):.4f}")
        print(f"F1 Score (weighted): {f1_score(all_labels, all_preds, average='weighted'):.4f}")
        print("Confusion Matrix:")
        print(confusion_matrix(all_labels, all_preds))
        print("Classification Report:")
        print(classification_report(all_labels, all_preds))

    _()

    return


if __name__ == "__main__":
    app.run()
