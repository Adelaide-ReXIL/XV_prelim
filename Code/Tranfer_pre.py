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
    from data_loader import XVData,csv_loader,PA_dataset_loader,bead_study_loader,sheep_dataset_loader
    from Triangle_Test import plot_3d_points
    import pandas as pd
    import random
    import numpy as np
    from sklearn.metrics import f1_score, confusion_matrix, classification_report


@app.cell
def _():
    import marimo as mo      

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return


@app.cell
def _():

    model=None
    if torch.cuda.is_available():
        model:PointNetPP = torch.load("pointnetpp_4d_full13.pth",weights_only=False)
    elif torch.mps.is_available():
        model: PointNetPP = torch.load("pointnetpp_4d_full8.pth", map_location=torch.device('mps'), weights_only=False)
    else:
        model: PointNetPP = torch.load("pointnetpp_4d_full8.pth", map_location=torch.device('cpu'), weights_only=False)




    model.classifier = nn.Sequential(
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, 2)
            )




    return (model,)


@app.cell
def _():

    data_files = []
    data_files.extend(csv_loader())
    data_files.extend(bead_study_loader())
    data_files.extend(PA_dataset_loader())

    wt = [d for d in data_files if 'wt' in d.lower()]
    cf = [d for d in data_files if 'wt'  not in d.lower()]
    wt.extend(sheep_dataset_loader())

    # Shuffle for randomness
    random.shuffle(wt)
    random.shuffle(cf)
    # Split each class into train and test (e.g., 80-20 split)
    wt_split = int(0.8 * len(wt))
    cf_split = int(0.8 * len(cf))

    wt_train, wt_test = wt[:wt_split], wt[wt_split:]
    cf_train, cf_test = cf[:cf_split], cf[cf_split:]

    train_files = wt_train + cf_train
    test_files = wt_test + cf_test

    random.shuffle(train_files)
    random.shuffle(test_files)

    label_map = {}

    for f in wt:
        label_map[f] = 0
    for f in cf:
        label_map[f] = 1

    print(len(wt),len(cf))

    return cf, data_files, label_map, test_files, train_files, wt


@app.cell
def _(data_files):
    def average_num_points(csv_files):
        total_points = 0
        for file in csv_files:
            df = pd.read_csv(file)
            total_points += len(df)
        return total_points / len(csv_files)
    average_num_points(data_files)
    return


@app.cell
def _(data_files):
    # Raw
    raw_df = pd.read_csv(data_files[0])
    raw_df.columns = ['SV', 'x', 'y', 'z']
    print(len(raw_df))
    plot_3d_points(raw_df)



    return


@app.cell
def _(data_files):
    # Sampled
    datas = XVData(data_files[:2], n=10000, transform=True,frames=False)
    sample = datas[0][0]
    sample_df = pd.DataFrame(sample.numpy(), columns=["SV", "x", "y", "z"])
    print(len(sample_df))
    plot_3d_points(sample_df)
    return


@app.cell
def _():
    return


@app.cell
def _(cf, label_map, model, train_files, wt):
    device = torch.device('cuda:0')
    model.to(device)


    dataset = XVData(train_files[:-16],n=7500,map=label_map)
    val = XVData(train_files[-16:],n=7500,map=label_map)


    weights = torch.tensor([1.0, len(cf)/len(wt)], device=device)
    criterion = nn.CrossEntropyLoss(weight=weights,label_smoothing=0.1)
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=3,  min_lr=1e-8)

    train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=28, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=4, shuffle=False, num_workers=28, pin_memory=True, drop_last=True)

    num_epochs = 75
    patience = 10
    best_val_loss = float('inf')
    counter = 0
    losses=[]
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4,  steps_per_epoch=len(train_loader), epochs=num_epochs)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for points, labels in train_loader:
            points = points.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).long()

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
                labels = labels.to(device, non_blocking=True).long()
                outputs = model(points)
                val_loss += criterion(outputs, labels).item()
        val_loss /= max(len(val_loader),1)

        scheduler.step(val_loss)


        losses.append(running_loss / len(train_loader))



        print(f"Epoch {epoch+1} Train Loss: {running_loss / len(train_loader):.4f}  Val Loss: {val_loss:.4f}")



    return device, losses


@app.cell
def _(losses):
    import matplotlib.pyplot as plt
    plt.plot(losses)
    return


@app.cell
def _(device, label_map, model, test_files):
    def _():
        import time

        test = XVData(test_files,n=10000,map=label_map)
        testloader = DataLoader(test, batch_size=2, shuffle=False, num_workers=16, pin_memory=True)

        model.eval()
        correct = total = 0

        with torch.no_grad():
            for points, labels in testloader:
                points = points.to(device, non_blocking=True).float()
                labels = labels.to(device, non_blocking=True).long()
                outputs = model(points)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        print(f"Test Accuracy: {100 * correct / total:.2f}%")

        model.eval()



        start = time.time()
        with torch.no_grad():
            output = model(points)
        end = time.time()

        pred = output
        print(f" Predicted: {pred}, Time: {(end - start) * 1000:.2f} ms")

        from sklearn.metrics import f1_score, confusion_matrix, classification_report

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for points, labels in testloader:
                points = points.to(device, non_blocking=True).float()
                labels = labels.to(device, non_blocking=True).long()
                outputs = model(points)
                preds = outputs.argmax(dim=1)

                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        # Concatenate all batches
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        acc = (all_preds == all_labels).mean() * 100
        print(f"Accuracy: {acc:.2f}%")

        # F1 Score
        print(f"F1 Score (macro): {f1_score(all_labels, all_preds, average='macro'):.4f}")
        print(f"F1 Score (weighted): {f1_score(all_labels, all_preds, average='weighted'):.4f}")

        # Confusion Matrix
        print("Confusion Matrix:")
        print(confusion_matrix(all_labels, all_preds))

        # Optional: Full report
        print("Classification Report:")
        print(classification_report(all_labels, all_preds))


    _()
    return


@app.cell
def _(model):
    torch.save(model, 'pretrained_16.pth')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
