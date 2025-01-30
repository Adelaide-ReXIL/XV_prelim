import pandas as pd
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler
import numpy as np
from scipy import stats
import math
import logging
from multiprocessing import Pool


def global_clustering_model(controls: list) -> KMeans:
    """
    Perform global clustering based on control samples, returns model that can be used in analysis

    Args:
        controls (list): A list of DataFrames of controls to be used for global clustering.

    Returns:
        KMeans: A KMeans model that can be used to cluster the data points with model.predict()
    """
    #for storing all specific ventilation values
    vent=[]
    for d in controls:
        d.columns=['Specific Ventilation (mL/mL)','x (mm)','y (mm)','z (mm)']
        vent.extend(d['Specific Ventilation (mL/mL)'])

    vent=np.array(vent)

    #model that makes global clusters
    model= KMeans(n_clusters = 6,init = 'k-means++')
    model.fit(vent.reshape(-1, 1))

    return model


def local_clustering(sample:pd.DataFrame)->pd.DataFrame:
    """
    Perform local clustering based on a sample DataFrame.

    Args:
        sample (pd.DataFrame): A DataFrame containing the XV data points to be clustered.

    Returns:
        pd.DataFrame: A DataFrame with an additional column 'lCluster' indicating the cluster assignment for each data point.
    """
    model= KMeans(n_clusters = 6,init = 'k-means++')
    cluster=model.fit_predict(pd.DataFrame(sample['Specific Ventilation (mL/mL)']))

    sample['lCluster']=cluster


    avg_vent=sample.groupby('lCluster')['Specific Ventilation (mL/mL)'].mean()
    sorted_clusters = avg_vent.sort_values().index


    cluster_mapping = {old: new for new, old in enumerate(sorted_clusters)}


    sample['lCluster'] = sample['lCluster'].map(cluster_mapping)


    return sample

def add_clusters(sample:pd.DataFrame,gModel:KMeans)->pd.DataFrame:
    """
    Perform clustering based on a sample DataFrame and a global clustering model.

    Args:
        sample (pd.DataFrame): A DataFrame containing the XV data points to be clustered.
        gModel (KMeans): A KMeans model obtained from global clustering.

    Returns:
        pd.DataFrame: A DataFrame with an additional column 'lCluster' indicating the local cluster assignment and 'gCluster'
        indicating global cluster for each data point.
    """
    #preform global clustering
    cluster=gModel.predict(pd.DataFrame(sample['Specific Ventilation (mL/mL)']))
    sample['gCluster']=cluster
    avg_vent=sample.groupby('gCluster')['Specific Ventilation (mL/mL)'].mean()
    sorted_clusters = avg_vent.sort_values().index


    cluster_mapping = {old: new for new, old in enumerate(sorted_clusters)}


    sample['gCluster'] = sample['gCluster'].map(cluster_mapping)

    #perform local clustering
    sample=local_clustering(sample)


    return sample

def extract_cluster_features(sample:pd.DataFrame,gModel:KMeans)->tuple[pd.DataFrame,np.ndarray]:
    """
    Extract features from a sample using the clustering method

    Args:
        sample (pd.DataFrame): A DataFrame containing the XV data points.
        gModel (KMeans): KMeans model to use for global clustering

    Returns:
        tuple[pd.DataFrame,pd.DataFrame]: A tuple of two DataFrames. The first DataFrame contains the XV data points with an additional
        column 'lCluster' indicating the local cluster assignment and 'gCluster' indicating global cluster for each data point.
        The second DataFrame contains the extracted features.
    """
    # Add global clusters to the sample DataFrame
    sample.columns=['Specific Ventilation (mL/mL)','x (mm)','y (mm)','z (mm)']
    sample = add_clusters(sample, gModel)



    #extract the features
    global_features = sample.groupby('gCluster')['Specific Ventilation (mL/mL)'].describe()[['mean']]     
    local_features = sample.groupby('lCluster')['Specific Ventilation (mL/mL)'].describe()[['mean']]     


    local_features.columns = ['l_' + col for col in local_features.columns]
    global_features.columns = ['g_' + col for col in global_features.columns]

    x_mean=sample['x (mm)'].mean()
    y_mean=sample['y (mm)'].mean()
    z_mean=sample['z (mm)'].mean()

    lower_right_front = sample[(sample['x (mm)'] > x_mean) & (sample['y (mm)'] < y_mean) & (sample['z (mm)'] > z_mean)]
    lower_right_back = sample[(sample['x (mm)'] > x_mean) & (sample['y (mm)'] < y_mean) & (sample['z (mm)'] < z_mean)]
    lower_left_front = sample[(sample['x (mm)'] < x_mean) & (sample['y (mm)'] < y_mean) & (sample['z (mm)'] > z_mean)]
    lower_left_back = sample[(sample['x (mm)'] < x_mean) & (sample['y (mm)'] < y_mean) & (sample['z (mm)'] < z_mean)]
    upper_right_front = sample[(sample['x (mm)'] > x_mean) & (sample['y (mm)'] > y_mean) & (sample['z (mm)'] > z_mean)]
    upper_right_back = sample[(sample['x (mm)'] > x_mean) & (sample['y (mm)'] > y_mean) & (sample['z (mm)'] < z_mean)]
    upper_left_front = sample[(sample['x (mm)'] < x_mean) & (sample['y (mm)'] > y_mean) & (sample['z (mm)'] > z_mean)]
    upper_left_back = sample[(sample['x (mm)'] < x_mean) & (sample['y (mm)'] > y_mean) & (sample['z (mm)'] < z_mean)]

    subsets = [
    ('lower_right_front', lower_right_front),
    ('lower_right_back', lower_right_back),
    ('lower_left_front', lower_left_front),
    ('lower_left_back', lower_left_back),
    ('upper_right_front', upper_right_front),
    ('upper_right_back', upper_right_back),
    ('upper_left_front', upper_left_front),
    ('upper_left_back', upper_left_back),
]

    # Initialize feature DataFrame
    local_features = local_features.reset_index()
    global_features = global_features.reset_index()
    features = pd.merge(local_features, global_features, left_on='lCluster', right_on='gCluster', how='outer')

    # Process each subset
    for name, subset in subsets:
        global_subset = subset.groupby('gCluster')['Specific Ventilation (mL/mL)'].describe()[[ '50%', '25%', '75%']]
        local_subset = subset.groupby('lCluster')['Specific Ventilation (mL/mL)'].describe()[[ '50%', '25%', '75%']]
        
        # Add prefixes for quadrant-specific features
        global_subset.columns = [f"{name}_g_" + col for col in global_subset.columns]
        local_subset.columns = [f"{name}_l_" + col for col in local_subset.columns]
        
        global_subset = global_subset.reset_index()
        local_subset = local_subset.reset_index()
        
        # Merge with main features DataFrame
        features = pd.merge(features, global_subset, on='gCluster', how='outer')
        features = pd.merge(features, local_subset, on='lCluster', how='outer')

    features=features.drop(columns=['gCluster','lCluster'])
    features = features.reset_index(drop=True)
    features=features.values.flatten()
    # features=np.append(features,extract_report(og_sample))


    return sample,features

def extract_features_report(sample:pd.DataFrame)->np.ndarray:
    """
    Extract features from a sample like the 4D medical reports in smaller fragments of the lung

    Args:
        sample (pd.DataFrame): A DataFrame containing the XV data points.

    Returns:
        pd.DataFrame: Data frame of features with each row corresponding to a part of the lung
        np.ndarray: A array of features flatten to 1-D from multidimensional matrix

    """

    x_mean=sample['x (mm)'].mean()
    y_mean=sample['y (mm)'].mean()
    z_mean=sample['z (mm)'].mean()

    lower_right_front = sample[(sample['x (mm)'] > x_mean) & (sample['y (mm)'] < y_mean) & (sample['z (mm)'] > z_mean)]
    lower_right_back = sample[(sample['x (mm)'] > x_mean) & (sample['y (mm)'] < y_mean) & (sample['z (mm)'] < z_mean)]
    lower_left_front = sample[(sample['x (mm)'] < x_mean) & (sample['y (mm)'] < y_mean) & (sample['z (mm)'] > z_mean)]
    lower_left_back = sample[(sample['x (mm)'] < x_mean) & (sample['y (mm)'] < y_mean) & (sample['z (mm)'] < z_mean)]
    upper_right_front = sample[(sample['x (mm)'] > x_mean) & (sample['y (mm)'] > y_mean) & (sample['z (mm)'] > z_mean)]
    upper_right_back = sample[(sample['x (mm)'] > x_mean) & (sample['y (mm)'] > y_mean) & (sample['z (mm)'] < z_mean)]
    upper_left_front = sample[(sample['x (mm)'] < x_mean) & (sample['y (mm)'] > y_mean) & (sample['z (mm)'] > z_mean)]
    upper_left_back = sample[(sample['x (mm)'] < x_mean) & (sample['y (mm)'] > y_mean) & (sample['z (mm)'] < z_mean)]

    subsets = [
    ('lower_right_front', lower_right_front),
    ('lower_right_back', lower_right_back),
    ('lower_left_front', lower_left_front),
    ('lower_left_back', lower_left_back),
    ('upper_right_front', upper_right_front),
    ('upper_right_back', upper_right_back),
    ('upper_left_front', upper_left_front),
    ('upper_left_back', upper_left_back),
]

    features=pd.DataFrame()
    for name, subset in subsets:
        newFeat=extract_report(subset)
        
        features = pd.concat([features,pd.DataFrame(newFeat)])




    return features,features.reset_index(drop=True).values.flatten()

def combine_features(cluster_features:pd.DataFrame,report_features:pd.DataFrame=pd.DataFrame(),single:bool=False):
    scaler=StandardScaler()
    label=cluster_features['Label']
    cluster_features=cluster_features.drop(columns=['Label'])
    cluster_features=pd.DataFrame(scaler.fit_transform(cluster_features.values))

    cluster_features['Label']=label
     
    if single:
        return cluster_features
    return pd.concat([cluster_features,pd.DataFrame(report_features)],axis=1)

    
def extract_report(df:pd.DataFrame):
    df.columns=['SV','X','Y','Z']
    mean=df['SV'].mean()
    median=df['SV'].median()
    vdp=len(df[df['SV']<0.6*mean])/len(df)
    



    kurtosis_value = stats.kurtosis(df['SV'])


    q1 = np.percentile(df['SV'], 25)
    q3 = np.percentile(df['SV'], 75)
    iqr = q3 - q1


    het=iqr/mean


    variance_value = np.var(df['SV'])

    return [mean,vdp,het,kurtosis_value,variance_value]




def neighbours(arr,i,j,k):
    
    binary_numbers = []

    binary_numbers.append(arr[i-1][j][k])

    binary_numbers.append(arr[i+1][j][k])

    binary_numbers.append(arr[i][j-1][k])

    binary_numbers.append(arr[i][j+1][k])

    binary_numbers.append(arr[i][j][k-1])

    binary_numbers.append(arr[i][j][k+1])

    return binary_numbers
        

def neighbours_4D(arr,t,i,j,k):
    
    binary_numbers = []

    binary_numbers.append(arr[t][i-1][j][k])

    binary_numbers.append(arr[t][i+1][j][k])

    binary_numbers.append(arr[t][i][j-1][k])

    binary_numbers.append(arr[t][i][j+1][k])

    binary_numbers.append(arr[t][i][j][k-1])

    binary_numbers.append(arr[t][i][j][k+1])

    if t-1>=0:
        binary_numbers.append(arr[t-1][i][j][k])
    else:
        binary_numbers.append(np.nan)

    if t+1<=14:
        binary_numbers.append(arr[t+1][i][j][k])
    else:
        binary_numbers.append(np.nan)

    return binary_numbers
        
def bin_to_dec(arr,i,j,k, neighbour_list):
    
    binary_result = [1 if num >= arr[i,j,k] else 0 for num in neighbour_list]

    binary_string = ''.join(map(str, binary_result))

    decimal_number = int(binary_string,2)
    
    return decimal_number

def bin_to_dec_4D(arr,t,i,j,k, neighbour_list):
    
    binary_result = [1 if num >= arr[t,i,j,k] else 0 for num in neighbour_list]

    binary_string = ''.join(map(str, binary_result))

    decimal_number = int(binary_string,2)
    
    return decimal_number

def compute_LBP_3D(grid):
    
    count = 0
    
    shape = grid.shape

    histogram = [0] * 64

    for i in range(1, shape[0]-1):

        for j in range(1, shape[1]-1):

            for k in range(1, shape[2]-1):

                neighbour_list = neighbours(grid,i,j,k)  # list of neighbours

                if math.isnan(grid[i,j,k]) or any(math.isnan(x) for x in neighbour_list): # ignore all nan values
                    continue

                decimal_number = bin_to_dec(grid,i,j,k,neighbour_list) # compute LBP and output a decimal number
                
                histogram[decimal_number] += 1
                
                count += 1  # count the total number of points contributing to LBP_3D
                
    return histogram, count

def compute_LBP_4D(grid):
    
    count = 0
    
    shape = grid.shape

    histogram = [0] * 256
    for t in range(1,shape[0]-1):
        for i in range(1, shape[1]-1):

            for j in range(1, shape[2]-1):

                for k in range(1, shape[3]-1):

                    neighbour_list = neighbours_4D(grid,t,i,j,k)  # list of neighbours

                    if math.isnan(grid[t,i,j,k]) or any(math.isnan(x) for x in neighbour_list): # ignore all nan values
                        continue

                    decimal_number = bin_to_dec_4D(grid,t,i,j,k,neighbour_list) # compute LBP and output a decimal number
                    
                    histogram[decimal_number] += 1  
                    
                    count += 1  # count the total number of points contributing to LBP_3D
                    
    return histogram, count


def create_grid(df, grid_size: tuple[int]):
    max_x, max_y, max_z = grid_size
    grid = np.full((max_x+1, max_y+1, max_z+1), np.nan)

    for _ , row in df.iterrows():

        value_column_name = 'SV'
        value = row[value_column_name]
        x = int(row['X'])
        y = int(row['Y'])
        z = int(row['Z'])


        if not np.isnan(value):
            grid[x, y, z] = value
            
    return grid

def create_grid_rank(df:pd.DataFrame):
    df['X']=df['X'].rank(method='dense').astype('int')-1
    df['Y']=df['Y'].rank(method='dense').astype('int')-1
    df['Z']=df['Z'].rank(method='dense').astype('int')-1
    max_x, max_y, max_z = df['X'].max(),df['Y'].max(),df['Z'].max()
    grid = np.full((max_x+1, max_y+1, max_z+1), np.nan)

    for _ , row in df.iterrows():

        value_column_name = 'SV'
        value = row[value_column_name]
        x = int(row['X'])
        y = int(row['Y'])
        z = int(row['Z'])


        if not np.isnan(value):
            grid[x, y, z] = value
            
    return grid

def create_grid_3D(df, grid_size: tuple[int]):
    df['X']=(df['X']+abs(df['X'].min()))//10
    df['Y']=(df['Y']+abs(df['Y'].min()))//10
    df['Z']=(df['Z']+abs(df['Z'].min()))//10
    max_x, max_y, max_z = df['X'].max(),df['Y'].max(),df['Z'].max()
    grid = np.full((14,int(np.ceil(max_x)+1),int(np.ceil( max_y)+1),int(np.ceil( max_z)+1)), np.nan)
    logging.debug('Initialising Empty Grid (Function)')

    for _ , row in df.iterrows():
        logging.debug('Initialising for a row (Function)')
        value_column_name = 'SV'
        value = row[value_column_name]
        x = int(round(row['X']))
        y = int(round(row['Y']))
        z = int(round(row['Z']))
        f=int(row['Frame'])

        if not np.isnan(value):
            if not np.isnan(grid[f,x,y,z]):
                grid[f,x,y,z]=(value+grid[f,x,y,z])/2
            else:
                grid[f,x, y, z] = value
                
    return grid
def create_grid_3D_rank(df):
    df['X']=df['X'].rank(method='dense').astype('int')-1
    df['Y']=df['Y'].rank(method='dense').astype('int')-1
    df['Z']=df['Z'].rank(method='dense').astype('int')-1
    max_x, max_y, max_z = df['X'].max(),df['Y'].max(),df['Z'].max()

    grid = np.full((14,max_x+1, max_y+1, max_z+1), np.nan)
    logging.debug('Initialising Empty Grid (Function)')

    for _ , row in df.iterrows():
        logging.debug('Initialising for a row (Function)')
        value_column_name = 'SV'
        value = row[value_column_name]
        x = int(row['X'])
        y = int(row['Y'])
        z = int(row['Z'])
        f=int(row['Frame'])

        if not np.isnan(value):
            if not np.isnan(grid[f,x,y,z]):
                grid[f,x,y,z]=(value+grid[f,x,y,z])/2
            else:
                grid[f,x, y, z] = value
                
    return grid

def create_grid_new(df):
    x_len=len(df['X'].unique())
    y_len=len(df['Y'].unique())
    z_val=sorted(df['Z'].unique())

    obj=[[[None]*len(z_val) for _ in range(y_len)] for _ in range(x_len)]
    obj=np.full((x_len,y_len,len(z_val)),np.nan)
    for i,x in enumerate(sorted(df['X'].unique())):
        for j,y in enumerate(sorted(df['Y'].unique())):
            mask = (df['X'] == x) & (df['Y'] == y)
            if any(mask):
                for k,z in enumerate(z_val):

                    m=(df['X'] == x) & (df['Y'] == y)&(df['Z']==z) 
                    if any(m):
                        obj[i,j,k]=df[m]['SV'].values[0]
    
    return obj

def conv_3d_patch(grid, scale=3): # scale = 3 for 3x3x3 patches
    shape = grid.shape
    new_grid = np.full(shape, np.nan)

    for i in range(shape[0] - 2):
        for j in range(shape[1] - 2):
            for k in range(shape[2] - 2):
                patch = grid[i:i+scale, j:j+scale, k:k+scale]
                if np.all(~np.isnan(patch)):
                    new_grid[i:i+scale, j:j+scale, k:k+scale] = patch
    return new_grid

# This function accept grid and new_grid as 3D array
def remain_percent(grid, new_grid):
    
    # Number of non-NaN in grid
    grid_not_nan = grid.size - np.sum(np.isnan(grid))

    # Number of non-NaN in new_grid
    new_grid_not_nan = new_grid.size - np.sum(np.isnan(new_grid))

    # percentage of remaining points
    percent = new_grid_not_nan/grid_not_nan

    return grid_not_nan, new_grid_not_nan, percent

# Convolution for extracting the removed points
def conv_remain_points(grid, new_grid):
    shape = grid.shape
    removed_points = np.full(shape, np.nan)

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                if grid[i,j,k] == new_grid[i,j,k]:
                    pass
                else:
                    removed_points[i,j,k] = grid[i,j,k]
                    
    return removed_points

class LBP_3D():
    def __init__(self,samples):
        self.samples=samples
        x,y,z=-1,-1,-1
        for s in samples:
            x=int(math.ceil(max(x,max(abs(s[0]['X'])))))
            y=int(math.ceil(max(y,max(abs(s[0]['Y'])))))
            z=int(math.ceil(max(z,max(abs(s[0]['Z'])))))
        
        self.gridSize=(x,y,z)
        self.features=pd.DataFrame()

    def extract(self)->pd.DataFrame:
        
        for s in self.samples:
            label=s[1]
            df=s[0]
            vol=abs((df['X'].max()-df['X'].min())*(df['Y'].max()-df['Y'].min())*(df['Z'].max()-df['Z'].min()))*10**-6
            df['SV']=(df['SV']-df['SV'].min())/(df['SV'].max()-df['SV'].min())*vol
            hist,count=compute_LBP_3D(create_grid_rank(df))
            feature=pd.DataFrame([np.array(hist)/count])
            feature['Label']=label
            self.features=pd.concat([self.features, pd.DataFrame(feature)], ignore_index=True)
        
        return self.features

class LBP_3DT():
    def __init__(self,samples):
        self.samples=samples
        x,y,z=-1,-1,-1
        for s in samples:
            x=int(math.ceil(max(x,max(abs(s[0]['X'])))))
            y=int(math.ceil(max(y,max(abs(s[0]['Y'])))))
            z=int(math.ceil(max(z,max(abs(s[0]['Z'])))))
        
        print(x,y,z)
        self.gridSize=(x,y,z)
        self.features=[]

    def extract(self)->pd.DataFrame:
        
        for s in self.samples:
            label=s[1]
            feat=pd.DataFrame()
            for i in range(14):
                df=s[0]
                vol=abs((df['X'].max()-df['X'].min())*(df['Y'].max()-df['Y'].min())*(df['Z'].max()-df['Z'].min()))*10**-6
                # df['SV']=(df['SV']-df['SV'].min())/(df['SV'].max()-df['SV'].min())
                grid=create_grid(df[df['Frame']==i],self.gridSize)
                hist,count=compute_LBP_3D(grid)
                feature=pd.DataFrame([np.array(hist)/count])
                feat=pd.concat([feat, pd.DataFrame(feature).T],axis=1, ignore_index=True)
            self.features.append((feat,label))
        
        return self.features
class LBP_4D():
    def __init__(self,samples):
        self.samples=samples
        x,y,z=-1,-1,-1
        for s in samples:
            x=int(math.ceil(max(x,max(abs(s[0]['X'])))))
            y=int(math.ceil(max(y,max(abs(s[0]['Y'])))))
            z=int(math.ceil(max(z,max(abs(s[0]['Z'])))))
        
        print(x,y,z)
        self.gridSize=(x,y,z)
        self.features=[]

    def extract(self)->pd.DataFrame:
        results = parallel_process(self.samples, self.gridSize, num_workers=4)

        for feature, label in results:
            self.features.append((feature, label))
        
        return self.features

def parallel_process(samples,grid_size,num_workers=4):
    with Pool(num_workers) as pool:
        results = pool.starmap(parallel_4D, [(sample, grid_size) for sample in samples])
    return results



def parallel_4D(s,grid_size):
    label=s[1]
    df=s[0]
    vol=abs((df['X'].max()-df['X'].min())*(df['Y'].max()-df['Y'].min())*(df['Z'].max()-df['Z'].min()))*10**-6
    df['SV']=(df['SV']-df['SV'].min())/(df['SV'].max()-df['SV'].min())*vol
    logging.debug('Initialising Gird for sample')
    grid=create_grid_3D_rank(df)
    logging.debug('Initialising Grid Complete')

    logging.debug('Performing 4D LBP')

    hist,count=compute_LBP_4D(grid)
    feature=pd.DataFrame([np.array(hist)/count])

    logging.debug('LBP for sample complete')
    return label,feature
class ClusterFeatures():

    def __init__(self,samples):
        self.samples=samples
        controls=[]
        for s in samples:
            if s[1]==0:
                controls.append(s[0])
        self.g_model=self.get_g_model(controls[:2])
        self.features=pd.DataFrame()


    def get_g_model(self,controls):
        return global_clustering_model(controls)

    def extract(self):

        for s in self.samples:

            _,feature=extract_cluster_features(s[0],self.g_model)
            feature=pd.DataFrame([feature])
            feature['Label']=s[1]
            self.features=pd.concat([self.features, pd.DataFrame(feature)], ignore_index=True)
        
        return self.features

        


    


if __name__=='__main__':
    import os
    import pickle

    logging.basicConfig(
        level=logging.DEBUG,  # Set the logging level to DEBUG
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log message format
    )


    
    mapping=pd.read_csv('../Datasets/XV Clinical Data/WCH_XV_genotypes.csv')
    mapping.columns=['Record ID','Condition']
    mapping=dict(zip(mapping['Record ID'],mapping['Condition']))


    csv_dir = '../Datasets/XV Clinical Data'
    csv_files = [fold for fold in os.listdir(csv_dir) if os.path.isdir(os.path.join(csv_dir, fold)) and 'WCH' in fold]
    dataframes=[]
    control=[]
    cf=[]
    for fold in csv_files:
        p=os.path.join(csv_dir, fold)
        key=int(p.split('-')[2])-10000
        files=[f for f in os.listdir(p) if os.path.isdir(os.path.join(p, f)) and '-LOBAR' not in f and 'WCH' in f]
        curr_csv=[]
        for f in files:
            path=os.path.join(p, f)
            csv=[pd.read_csv(os.path.join(path,c)) for c in os.listdir(path) if c.endswith('_final.csv')]
            if 'EXP' in f:
                csv[0]['Frame']=csv[0]['Frame']+7
            curr_csv.extend(csv)
        

        dataframes.append(pd.merge(curr_csv[0], curr_csv[1], how='outer'))
        if 'Control' in mapping[key]:
            control.append(pd.merge(curr_csv[0], curr_csv[1], how='outer'))
        else:
            cf.append(pd.merge(curr_csv[0], curr_csv[1], how='outer'))
        
    dataframes=[]
    for c in control:
        c.columns=['Frame','SV','X','Y','Z']
        dataframes.append([c,0])
    for c in cf:
        c.columns=['Frame','SV','X','Y','Z']
        dataframes.append([c,1])

    lbp=LBP_4D(dataframes)
    features=lbp.extract()
    with open('lbp2R','wb') as fp:

        pickle.dump(features,fp)
        logging.debug('Done writing file')

    csv_dir = '../Datasets/XV Clinical Data/adult_controls_from_Miami'
    csv_files = [fold for fold in os.listdir(csv_dir) if os.path.isdir(os.path.join(csv_dir, fold))]


    dataframe=[]

    for fold in csv_files:
        p=os.path.join(csv_dir, fold)
        files=[f for f in os.listdir(p) if os.path.isdir(os.path.join(p, f)) and '-LOBAR' not in f]
        curr_csv=[]
        for f in files:
            path=os.path.join(p, f)
            csv=[pd.read_csv(os.path.join(path,c)) for c in os.listdir(path) if c.endswith('_final.csv')]
            if 'EXP' in f:
                csv[0]['Frame']=csv[0]['Frame']+7
            curr_csv.extend(csv)
        

        dataframe.append(pd.merge(curr_csv[0], curr_csv[1], how='outer'))

    dataframes=[]
    for c in dataframe:

        c.columns=['Frame','SV','X','Y','Z']
        dataframes.append([c,0])

    lbp=LBP_4D(dataframes)
    features=lbp.extract()
    with open('lbpAdR','wb') as fp:

        pickle.dump(features,fp)
        logging.debug('Done writing file')




