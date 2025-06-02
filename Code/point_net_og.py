import marimo

__generated_with = "0.13.11"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    return


@app.cell
def _(F, nn, torch):
    class PointNetPP(nn.Module):
        def __init__(self, input_dim=4, num_classes=2):
            super().__init__()

            self.sa1 = PointNetSetAbstraction(
                npoint=512,
                mlp=[input_dim, 64, 64, 128]
            )

            self.sa2 = PointNetSetAbstraction(
                npoint=128,
                mlp=[128, 128, 128, 256]
            )

            self.sa3 = PointNetSetAbstraction(
                npoint=None,
                mlp=[256, 256, 512, 1024]
            )

            self.fc1 = nn.Linear(1024, 512)
            self.bn1 = nn.BatchNorm1d(512)
            self.drop1 = nn.Dropout(0.4)

            self.fc2 = nn.Linear(512, 256)
            self.bn2 = nn.BatchNorm1d(256)
            self.drop2 = nn.Dropout(0.4)

            self.fc3 = nn.Linear(256, num_classes)

        def forward(self, input):
            B, N, _ = input.shape
            coords = input[:, :, 1:]  # x,y,z
            features = input  # s,x,y,z

            l1_coords, l1_features = self.sa1(coords, features)
            l2_coords, l2_features = self.sa2(l1_coords, l1_features)
            l3_coords, l3_features = self.sa3(l2_coords, l2_features)

            x = l3_features.view(B, -1)
            x = self.drop1(F.relu(self.bn1(self.fc1(x))))
            x = self.drop2(F.relu(self.bn2(self.fc2(x))))
            x = self.fc3(x)

            return x

    class PointNetSetAbstraction(nn.Module):
        def __init__(self, npoint, mlp):
            super().__init__()
            self.npoint = npoint
            layers = []
            for i in range(len(mlp) - 1):
                layers.append(nn.Conv1d(mlp[i], mlp[i+1], 1))
                layers.append(nn.BatchNorm1d(mlp[i+1]))
                layers.append(nn.ReLU())
            self.mlp = nn.Sequential(*layers)

        def forward(self, coords, features):
            B, N, _ = coords.shape
            if self.npoint is not None:
                if N < self.npoint:
                    raise ValueError(f"N={N} is less than npoint={self.npoint}")
                idx = torch.stack([torch.randperm(N, device=coords.device)[:self.npoint] for _ in range(B)])
                new_coords = torch.gather(coords, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
                new_features = torch.gather(features, 1, idx.unsqueeze(-1).expand(-1, -1, features.shape[-1])).permute(0, 2, 1)
            else:
                new_coords = coords[:, :1, :]
                new_features = features.permute(0, 2, 1)
            new_features = self.mlp(new_features)
            new_features = torch.max(new_features, 2)[0].unsqueeze(2).repeat(1, 1, new_coords.shape[1])
            return new_coords, new_features.permute(0, 2, 1)

    return


if __name__ == "__main__":
    app.run()
