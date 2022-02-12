"""# Build Dataset"""
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

import numpy as np


class BuildDataset:
    def __init__(self):
        self.chunk_size = 10000
        self.df = pd.read_csv('jena_climate_2009_2016.csv')
    
    
    def df_to_np(self):
        '''
        Convert pandas dataframe into numpy array.
        
        '''
        features_considered = ['p (mbar)', 'T (degC)', 'rho (g/m**3)', 
                               'Tpot (K)', 'Tdew (degC)', 'rh (%)', 
                               'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 
                               'sh (g/kg)', 'H2OC (mmol/mol)', 'wv (m/s)', 
                               'max. wv (m/s)', 'wd (deg)']
        self.target_feature = 'T (degC)'
    
        features = self.df[features_considered]
        self.dataset = features.values     
    
    
    def get_dl(self, args):
        '''
        Create dataloaders
        
        '''
        self.df_to_np()
        
        # This dataset can be divided into 42 chunks of len 10000, plus
        # one chunk of size<10000. So, 43 chunks in total.
        num_chunks = (self.dataset.shape[0] // self.chunk_size) + 1 # 43
        
        print('\n***\nPreparing Dataset...')
        x_list, y_list = [], []
        for i in range(num_chunks):
            chunk = self.dataset[i:i+self.chunk_size]
            x, y = transform(chunk)

            x_list.append(x)
            y_list.append(y)
            
            if i%10 == 0:
                percentage_processed = i/num_chunks * 100
                print(f'{percentage_processed:.0f}%')
        
        print('100%')
    
        # Validation set
        random_idx = np.random.randint(0, len(y_list))
        x_val, y_val = x_list.pop(random_idx), y_list.pop(random_idx)
        val_ds = TensorDataset(x_val, y_val)
        val_dl = DataLoader(val_ds, 
                            batch_size=64, 
                            shuffle=True)

        # Test set
        random_idx = np.random.randint(0, len(y_list))
        x_test, y_test = x_list.pop(random_idx), y_list.pop(random_idx)       
        test_ds = TensorDataset(x_test, y_test)
        test_dl = DataLoader(test_ds, 
                             batch_size=64, 
                             shuffle=True)
                
        # Training set
        x_list, y_list = torch.cat(x_list, dim=0), torch.cat(y_list, dim=0)
        train_ds = TensorDataset(x_list, y_list)
        train_dl = DataLoader(train_ds, 
                              batch_size=args.batch_size, 
                              shuffle=True)
        
        # Store some stuff for the function 'sample_predictions'
        self.val_ds = val_ds
        self.x_val, self.y_val = x_val, y_val
        
        self.test_ds = test_ds
        self.x_test, self.y_test = x_test, y_test
        
        return train_dl, val_dl, test_dl
    
    
def transform(chunk, target_size=72):
    '''
    The original sequence length is 720, which is converted into 120 because 
    the temperature does not change significantly every hour.
    
    '''
    target_idx = 1 # index of 'T (degC)' is 1
    target = chunk[:, target_idx] 
    
    data, labels = [], []
    start_idx = 720
    end_idx = len(chunk) - target_size + 1
    
    for i in range(start_idx, end_idx):
        indices = range(i - start_idx, i, 6)
        data.append(chunk[indices])
        labels.append(target[i : i + target_size])
    
    # Convert lists into numpy arrays first because converting lists directly
    # to tensors is very slow. 
    data, labels = np.array(data), np.array(labels)
    x, y = torch.FloatTensor(data), torch.FloatTensor(labels)

    return x, y