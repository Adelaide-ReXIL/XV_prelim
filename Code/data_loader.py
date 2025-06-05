import marimo

__generated_with = "0.13.11"
app = marimo.App(width="full")

with app.setup:
    # Initialization code that runs before all other cells
    from normalised_sv import rotate_lung_data
    from torch.utils.data import Dataset
    import pandas as pd
    import numpy as np
    from augumented_lung import aug_lung
    import torch
    import os


@app.cell
def _():
    import marimo as mo
    return


@app.function
def find_max_min(files):
    data=np.array([])
    for f in files:
        temp=pd.read_csv(f)
        if len(temp.columns)==5:
            temp=temp[temp['Frame'] == 6].drop(columns=['Frame'])
            
        temp.columns=['SV','x','y','z']
        data=np.concatenate((data,temp['SV'].values))

    return data.min(),data.max()


@app.class_definition
class XVData(Dataset):
    def __init__(self, csv_files, transform=False,n=6000,frames=False,map={}):
        self.csv_files = csv_files
        self.transform = transform
        self.n=n
        self.sv_norm=find_max_min(csv_files)
        self.frames=frames
        self.label=map

    def __len__(self):
        return len(self.csv_files)

    def _get_label(self, name):
        if self.label:
            return self.label[name]        
        if ('WT' in name) or ( 'wt' in name):
            return 0
        else:
            return 1
        

    def __getitem__(self, idx):
        data = pd.read_csv(self.csv_files[idx])
        if self.frames==True or len(data.columns)==5:
            data=data[data['Frame'] == 6].drop(columns=['Frame'])
        data.columns = ['SV', 'x', 'y', 'z']

        if self.transform:
            data = rotate_lung_data(data)

        data=aug_lung(data,int((self.n+5000)/len(data)))

        pts = data[['x', 'y', 'z']].values
        s = data[['SV']].values
        s=(s-self.sv_norm[0])/(self.sv_norm[1]-self.sv_norm[0]+1e-8)

        mins = pts.min(0, keepdims=True)
        maxs = pts.max(0, keepdims=True)
        norm_pts = (pts - mins) / (maxs - mins + 1e-8)

        pts_out = np.hstack([s, norm_pts])

        if len(pts_out) >= self.n:
            indices = np.random.choice(len(pts_out), self.n, replace=False)
        else:
            indices = np.random.choice(len(pts_out), self.n, replace=True)

        pts_out = pts_out[indices]
        label = self._get_label(self.csv_files[idx])

        return torch.tensor(pts_out, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


@app.function
def PA_dataset_loader(dir='../Datasets/Rat PA Study/'):
    csv_dir = os.path.join(dir, 'csv')
    files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith('.csv') and 'PA7' in f]
    return files


@app.function
def sheep_dataset_loader(dir='../Datasets/Output'):
    mapping = pd.read_csv(dir+'/sheep_ids_types.csv')
    id_mapping=dict(zip(mapping['ID'],mapping['Challenge']))
    preg_mapping=dict(zip(mapping['ID'],mapping['U/S pregnancy']))
    dir = os.path.join(dir, 'Specific Ventilation')
    dirs = [os.path.join(dir, d) for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]

    files=[]

    for d in dirs:
        name=d.split('/')[-1]
        if "B" in name:
            continue
        if id_mapping.get(name[:-2]," ") not in ["Control"," "]:
            continue
        if preg_mapping.get(name[:-2]," ") not in ["Non-pregnant"," "]:
            continue
        files.append(f"{d}/output/{name}.specificVentilation.insp.pp.07.csv")


    return files


@app.function
def csv_loader(csv_dir='../Datasets/csv'):
    files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith('.csv')]
    return files


@app.function
def bead_study_loader(dir='../Datasets/Rat Sterile Bead Study'):
    dir+='/csv'
    files=[]
    files.extend(csv_loader(dir+"/baseline"))
    files.extend(csv_loader(dir+"/Control"))
    return files


@app.cell
def _():
    find_max_min(sheep_dataset_loader())
    return


@app.function
def child_loader(dir='../Datasets/XV Clinical Data'):
    mapping=pd.read_csv(f'{dir}/WCH_XV_genotypes.csv')
    mapping = mapping.iloc[:, :-2]
    mapping.columns = ['Record ID', 'Condition']
    mapping=dict(zip(mapping['Record ID'],mapping['Condition']))
    csv_files = [fold for fold in os.listdir(dir) if os.path.isdir(os.path.join(dir, fold)) and 'WCH' in fold]

    control=[]
    cf=[]
    for fold in csv_files:
        p=os.path.join(dir, fold)
        key=int(p.split('-')[2])-10000
        if key==26: key=25
        files=[f for f in os.listdir(p) if os.path.isdir(os.path.join(p, f)) and '-LOBAR' not in f and 'WCH' in f]
        curr_csv=[]
        for f in files:
            path=os.path.join(p, f)
            csv = [c for c in os.listdir(path) if c.endswith('_final.csv') and 'INSP' in c]
            if csv:
                csv=csv[0]
            else:
                continue
                
            if 'Control' in mapping[key]:
                control.append(path+'/'+csv)
            else:
                cf.append(path+'/'+csv)
            


    return control,cf


@app.function
def adult_loader(dir='../Datasets/XV Clinical Data/adult_controls_from_Miami'):
    csv_files = [fold for fold in os.listdir(dir) if os.path.isdir(os.path.join(dir, fold))]
    control=[]
    for fold in csv_files:
        p=os.path.join(dir, fold)
        files=[f for f in os.listdir(p) if os.path.isdir(os.path.join(p, f)) and '-LOBAR' not in f]
        curr_csv=[]
        for f in files:
            path=os.path.join(p, f)
            csv = [c for c in os.listdir(path) if c.endswith('_final.csv') and 'INSP' in c]
            if csv:
                csv=csv[0]
            else:
                continue
            control.append(path+'/'+csv)

    return control


if __name__ == "__main__":
    app.run()
