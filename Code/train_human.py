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
        model:PointNetPP = torch.load('pretrained_16.pth',weights_only=False)
    elif torch.mps.is_available():
        model: PointNetPP = torch.load('pretrained_best_86.pth', map_location=torch.device('mps'), weights_only=False)
    else:
        model: PointNetPP = torch.load('pretrained_best_86.pth', map_location=torch.device('cpu'), weights_only=False)

    # for block in [model.sa2,model.sa3]:
    #     for param in block.parameters():
    #         param.requires_grad = False


    model




    return (model,)


@app.cell
def _():
    control, cf = child_loader()
    control.extend(adult_loader())

    random.shuffle(control)
    random.shuffle(cf)

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
def _(label_map, model, train_files):
    device = torch.device('cuda:0')
    model.to(device)


    dataset = XVData(train_files,n=7500,frames=True,map=label_map)
    losses=[]


    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=3,  min_lr=1e-8)

    train_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=28, pin_memory=True, drop_last=True)
    num_epochs = 100
    patience = 10
    best_val_loss = float('inf')
    counter = 0

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4,  steps_per_epoch=len(train_loader), epochs=num_epochs)
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
            scheduler.step()





        # scheduler.step(val_loss)

        losses.append(running_loss / len(train_loader))
        print(f"Epoch {epoch+1} Train Loss: {running_loss / len(train_loader):.4f}")



    return device, losses, train_loader


@app.cell
def _(losses):
    import matplotlib.pyplot as plt
    plt.plot(losses)
    return


@app.cell
def _():
    return


@app.cell
def _(device, label_map, model, test_files, train_loader):
    def _():
        import time

        test = XVData(test_files,n=7500,frames=True,map=label_map)
        testloader = DataLoader(test, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)

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

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for points, labels in train_loader:
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
    torch.save(model, 'trained_8.pth')
    return


@app.cell
def _():
    a,b=child_loader()
    return a, b


@app.cell
def _(a, b, device, model, train_loader):
    def _():
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for points, labels in train_loader:
                points = points.to(device, non_blocking=True).float()
                labels = labels.to(device, non_blocking=True).long()
                outputs = model(points)
                preds = outputs.argmax(dim=1)

                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        # Concatenate all batches
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        print(all_preds,all_labels)
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

        all_preds = []
        all_labels = []
        model.eval()
        new_map={}

        for i in b:
            new_map[i]=1
        for i in a:
            new_map[i]=0



        a.extend(b)
        set=XVData(a,frames=True,n=7500,map=new_map)
        loader=DataLoader(set , batch_size=1, shuffle=False, num_workers=28, pin_memory=True, drop_last=True)
        with torch.no_grad():
            for points, labels in loader:
                points = points.to(device, non_blocking=True).float()
                labels = labels.to(device, non_blocking=True).long()
                outputs = model(points)
                preds = outputs.argmax(dim=1)

                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        # Concatenate all batches
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        print(all_preds,all_labels)
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


if __name__ == "__main__":
    app.run()
