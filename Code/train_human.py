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


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    model=None
    if torch.cuda.is_available():
        model:PointNetPP = torch.load('pretrained_13.pth',weights_only=False)
    elif torch.mps.is_available():
        model: PointNetPP = torch.load('pretrained_best_86.pth', map_location=torch.device('mps'), weights_only=False)
    else:
        model: PointNetPP = torch.load('pretrained_best_86.pth', map_location=torch.device('cpu'), weights_only=False)

    model




    return (model,)


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

    return label_map, test_files, train_files


@app.cell
def _(label_map, model, train_files):
    device = torch.device('cuda:0')
    model.to(device)


    dataset = XVData(train_files,n=7500,frames=True,map=label_map)
    losses=[]


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=3,  min_lr=1e-8)

    train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=28, pin_memory=True, drop_last=True)
    num_epochs = 50
    patience = 10
    best_val_loss = float('inf')
    counter = 0

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


        # scheduler.step(val_loss)

        losses.append(running_loss / len(train_loader))
        print(f"Epoch {epoch+1} Train Loss: {running_loss / len(train_loader):.4f}")



    return device, losses


@app.cell
def _(losses):
    import matplotlib.pyplot as plt
    plt.plot(losses)
    return


@app.cell
def _():
    return


@app.cell
def _(device, label_map, model, test_files):
    def _():
        import time

        test = XVData(test_files,n=7500,frames=True,map=label_map)
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
    torch.save(model, 'trained_5.pth')
    return


if __name__ == "__main__":
    app.run()
