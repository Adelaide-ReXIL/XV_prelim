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

    return df, sample


@app.cell
def _(df):
    plot_3d_points(df)
    return


@app.function
def ball_query(coords, centers, radius, max_samples):
    B, N, _ = coords.shape
    S = centers.shape[1]

    dists = torch.cdist(centers, coords)  # (B, S, N)
    mask = dists <= radius  # (B, S, N)

    # Fill far distances with large value for argsort
    dists[~mask] = 1e10

    # Get top-k closest valid neighbors (argsort + narrow)
    sorted_idx = dists.argsort(dim=-1)[:, :, :max_samples]  # (B, S, max_samples)
    sorted_dists = torch.gather(dists, 2, sorted_idx)       # (B, S, max_samples)

    valid_mask = sorted_dists < 1e10                        # (B, S, max_samples)

    return sorted_idx, valid_mask


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
def _(sample):
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



    pos = sample[:, 1:]  # [N, 3]
    coords = pos.unsqueeze(0)  # [1, N, 3]
    centers = geometry_centers(coords, num_segments=8)  # [1, S, 3], where S = 6×6×6
    centers_flat = centers.view(1, 8**3, 3)

    neighbor_indices, valid_mask = ball_query(coords, centers_flat, radius=0.5, max_samples=256)


    visualize_ball_query(coords, centers_flat, neighbor_indices, valid_mask)


    return


@app.class_definition
class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, mlp, radius=0.5, max_samples=32, use_geometry_centers=False, num_segments=6, axis=2):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.max_samples = max_samples
        self.use_geometry_centers = use_geometry_centers
        self.num_segments = num_segments
        self.axis = axis

        layers = []
        in_channel = mlp[0]
        for out_channel in mlp[1:]:
            layers += [nn.Conv1d(in_channel, out_channel, 1), nn.BatchNorm1d(out_channel), nn.ReLU()]
            in_channel = out_channel
        self.mlp = nn.Sequential(*layers)

    def forward(self, coords, features):
        B, N, _ = coords.shape

        if self.npoint is not None:
            if self.use_geometry_centers:
                centers = geometry_centers(coords, self.num_segments).view(B, -1, 3)
                S = centers.shape[1]
            else:
                idx = torch.randint(N, (B, self.npoint), device=coords.device)
                centers = coords.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))
                S = self.npoint

            neighbor_idx, valid_mask = ball_query(coords, centers, self.radius, self.max_samples)

            batch_idx = torch.arange(B, device=coords.device).view(B, 1, 1).expand(B, S, self.max_samples)
            neighbor_coords = coords[batch_idx, neighbor_idx]
            neighbor_features = features[batch_idx, neighbor_idx]

            rel_coords = neighbor_coords - centers.unsqueeze(2)
            combined = torch.cat((rel_coords, neighbor_features), dim=-1)
            combined *= valid_mask.unsqueeze(-1).float()

            x = combined.permute(0, 1, 3, 2).reshape(B * S, -1, self.max_samples)
            x = self.mlp(x)
            x = torch.max(x, 2)[0].view(B, S, -1)

            return centers, x

        x = self.mlp(features.permute(0, 2, 1))
        x = torch.max(x, 2, keepdim=True)[0].permute(0, 2, 1)
        return coords[:, :1], x


@app.class_definition
class PointNetPP(nn.Module):
    def __init__(self, input_dim=4, num_classes=2):
        super().__init__()

        self.sa1 = PointNetSetAbstraction(
            npoint=512,
            mlp=[input_dim + 3, 64, 64, 128],
            max_samples=128,
            num_segments=8,
            use_geometry_centers=True
        )

        self.sa2 = PointNetSetAbstraction(
            npoint=128,
            mlp=[128 + 3, 128, 128, 256],
            max_samples=64,
            num_segments=6,
            radius=0.25,
            use_geometry_centers=True
        )

        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            mlp=[256, 256, 512, 1024],
            max_samples=32,
            num_segments=2,
            radius=0.1
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
        coords = x[:, :, 1:]
        l1_coords, l1_feats = self.sa1(coords, x)
        l2_coords, l2_feats = self.sa2(l1_coords, l1_feats)
        _, l3_feats = self.sa3(l2_coords, l2_feats)

        return self.classifier(l3_feats.view(x.shape[0], -1))


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
    optimizer = optim.Adam(model.parameters(), lr=5e-5)

    num_epochs = 200
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break
    return (model,)


@app.cell
def _(model):
    torch.save(model, 'pointnetpp3_full.pth')
    return


@app.cell
def _():
    def _():
        from sklearn.metrics import f1_score, confusion_matrix, classification_report
        import time

        device = torch.device('cuda:0')
        model = torch.load('pointnetpp3_full.pth', weights_only=False)
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
