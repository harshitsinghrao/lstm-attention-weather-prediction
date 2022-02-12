import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import namedtuple

from neural_network import Encoder, GlobalAttn, Decoder
from build_dataset import BuildDataset
from evaluate import evaluate_performance, sample_predictions


class Arguments:
    ''' A class for storing arguments '''
    pass

if __name__ == '__main__':
    args = Arguments()
    
    use_cuda = torch.cuda.is_available()
    print(f'CUDA {use_cuda}')
    args.device = torch.device('cuda' if use_cuda else 'cpu')
    
    # Total number of features considered like 'p (mbar)', 'T (degC)', 
    # 'rho (g/m**3)', etc. in order to predict the target feature.   
    args.num_features = 14
    # Total number of timesteps to be predicted.
    args.target_size = 72
    
    # Hyper params
    args.lstm1_size = 256 
    args.lstm2_size = 128 
    args.lstm_size = args.lstm2_size
    args.fc_size = 16
    args.lr = 5e-4
    args.max_lr = 4e-4
    args.batch_size = 256
    
    # Build the dataset.
    build_ds = BuildDataset()
    train_dl, val_dl, test_dl = build_ds.get_dl(args)
    
    # Initialize the neural network
    NeuralNetwork = namedtuple('NeuralNetwork', 'encoder, attention, decoder')
    model = NeuralNetwork(encoder = Encoder(args).to(args.device), 
                          attention = GlobalAttn(args).to(args.device), 
                          decoder = Decoder(args).to(args.device))

    # Mean Absolute Error
    criterion = nn.MSELoss() 
    
    params = (list(model.encoder.parameters()) + 
              list(model.attention.parameters()) + 
              list(model.decoder.parameters()))
    optimizer = optim.Adam(params, lr=args.lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr = 1e-4, 
                                            max_lr = 10**(-3.7), 
                                            step_size_up = 3*1560, 
                                            gamma=0.99994, 
                                            cycle_momentum=False)
    
    # Keep track of losses for graph plotting.
    train_losses = []
    val_losses = []
    
    args.epochs = 40 
    print('\n=====\nTRAINING LOOP')
    for epoch in range(args.epochs):
        print(f'\n***\nEPOCH: {epoch+1}/{args.epochs}')
        
        # Training
        print('\nTraining Losses:')
        model.encoder.train()
        model.attention.train()
        model.decoder.train()        
    
        for i, (x, y) in enumerate(train_dl):
            x, y = x.to(args.device), y.to(args.device)
            
            hs_bar, hS = model.encoder(x)
            pred = model.decoder(model.attention, hs_bar, hS)
    
            loss = criterion(pred, y).to(args.device)
            loss.backward()
    
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    
            if i % 200 == 0:
                print(f'{loss.item():.4f}')
            if i % 10 == 0:
                train_losses.append(loss.detach().cpu().numpy())
    
        # Validation
        print('\nValidation Set Losses:')
        epoch_val_losses = evaluate_performance(val_dl, model, criterion, args)
        val_losses += epoch_val_losses
        
        if epoch % 5 == 0:
            # Sample some predictions from the validation set
            x_val = build_ds.x_val
            val_ds = build_ds.val_ds
            sample_predictions(model, criterion, x_val, val_ds, args)
        
            # Plot training and validation losses
            plt.plot(val_losses, 'c', label='Validation Loss')
            plt.plot(np.clip(train_losses, a_min=None, a_max=5), 
                     label='Training Loss')
            plt.legend(loc='upper left')
            plt.show()
    
    # Test set
    print('\n=====\nTest Set Losses:')
    _ = evaluate_performance(test_dl, model, criterion, args)
    
    # Sample some predictions from the test set
    x_test = build_ds.x_test
    test_ds = build_ds.test_ds
    sample_predictions(model, criterion, x_test, test_ds, args)