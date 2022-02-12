import numpy as np
import torch
import matplotlib.pyplot as plt

def evaluate_performance(dl, model, criterion, args):
    '''
    Evaluate current performance of the model by calculating loss for 
    the validation and test sets.
    
    Here, dl is the dataloader.
    
    '''
    losses = []
    
    model.encoder.eval()
    model.attention.eval()
    model.decoder.eval()
    
    with torch.no_grad():
        for i, (x, y) in enumerate(dl):
            x, y = x.to(args.device), y.to(args.device)

            hs_bar, hS = model.encoder(x)
            pred = model.decoder(model.attention, hs_bar, hS)

            loss = criterion(pred, y).to(args.device)
        
            if i%40 == 0:
                print(f'{loss.item():.5f}')
            losses.append(loss.cpu().numpy())
            
    avg_loss = np.mean(losses)
    print(f'\nAverage MSE: {avg_loss:.4f}')
    
    return losses


def plot_sample_predictions(history, true_future, prediction):
    '''
    Graph plotting for the function sample_predictions.
    
    '''
    history = np.squeeze(history)
    true_future = np.squeeze(true_future)
    prediction = np.squeeze(prediction)

    plt.plot(np.arange(-len(history),0), 
             np.array(history[:, 1]), 
             label='History')
    
    plt.plot(np.arange(len(true_future)), 
             np.array(true_future), 'b', 
             label='True Future')
    
    plt.plot(np.arange(len(true_future)), 
             np.array(prediction), 'r', 
             label='Predicted Future')
    
    plt.legend(loc='upper left')
    plt.show()


def sample_predictions(model, criterion, x_data, ds, args):
    '''
    Sample some predictions from the validation and test sets. 
    
    ds: PyTorch Dataset.
    x_data: the x part of the data. 
    
    '''
    print('\n***\nSample Some Predictions:')
    num_samples = x_data.shape[0]
    
    model.encoder.eval()
    model.attention.eval()
    model.decoder.eval()
    
    with torch.no_grad():
        for _ in range(3):
            idx = np.random.randint(0, num_samples)
            x, y = ds[idx]
            x = torch.unsqueeze(x,0)
            y = torch.unsqueeze(y,0)
            x, y = x.to(args.device), y.to(args.device)
    
            enc_h, hS = model.encoder(x)
            pred = model.decoder(model.attention, enc_h, hS)
            
            loss = criterion(pred, y).to(args.device)
    
            x, y, pred = x.cpu().numpy(), y.cpu().numpy(), pred.cpu().numpy()
            plot_sample_predictions(x, y, pred)
            print(f'\nloss: {loss.item():.5f}\n\n***')