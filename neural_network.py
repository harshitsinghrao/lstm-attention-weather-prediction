import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.bn0 = nn.BatchNorm1d(args.num_features)

        self.conv1 = nn.Conv1d(args.num_features, 64, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.conv3 = nn.Conv1d(64, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.conv4 = nn.Conv1d(128, 128, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        
        self.lstm1 = nn.LSTM(128, args.lstm1_size)
        self.bn5 = nn.BatchNorm1d(args.lstm1_size)
        self.drop = nn.Dropout()
        
        self.lstm2 = nn.LSTM(args.lstm1_size, args.lstm2_size)
        
    def forward(self, x):
        # (batch, seq_len, in_size)
        x = x.permute(0,2,1) # (batch, channels, length)
        x = self.bn0(x) # (batch, channels, length)
        
        x = self.conv1(x) # (batch, channels, length)
        x = self.bn1(x) # same shape
        x = F.relu(x) # same shape
        
        x = self.conv2(x)# (batch, channels, length)
        x = self.bn2(x) # same shape
        x = F.relu(x)# same shape
        
        x = F.max_pool1d(x,2) # (batch, channels, length/2)
        
        x = self.conv3(x) # (batch, channels, length/2)
        x = self.bn3(x) # same shape
        x = F.relu(x)  # same shape
        
        x = self.conv4(x) # (batch, channels, length/2)
        x = self.bn4(x) # same shape
        x = F.relu(x) # same shape
        
        x = F.max_pool1d(x,2) # (batch, channels, length/4)
        
        # change of notation: (batch, in_features, seq_len)
        x = x.permute(2,0,1) # (seq_len, batch, in_features)
        x, _ = self.lstm1(x) # (seq_len, batch, lstm1_size)
        x = x.permute(1,2,0) # (batch, lstm1_size, seq_len)
        x = self.bn5(x) # same shape
        x = self.drop(x) # same shape
        x = F.relu(x) # same shape
        
        x = x.permute(2,0,1) # (seq_len, batch, lstm1_size)
        hs_bar, (hS,_) = self.lstm2(x) 
        # x (seq_len, batch, lstm2_size), hS (1, batch, lstm2_size)
        return hs_bar, hS


class GlobalAttn(nn.Module):
    def __init__(self, args):
        super(GlobalAttn, self).__init__()
        # lstm_size is the size of top layer of encoder
        self.W = nn.Linear(args.lstm_size, args.lstm_size)
     
    def score(self, hs_bar, ht):
        '''
        The score function is the general score. 
        
        Returns:
            score: of shape (batch, seq_len, 1)
        '''
        # (batch, lstm_size)
        ht = torch.unsqueeze(ht, dim=1) # (batch, 1, lstm_size)
        ht = torch.transpose(ht,1,2) # (batch, lstm_size, 1)

        # (seq_len, batch, lstm_size)
        hs_bar = torch.transpose(hs_bar,0,1) # (batch, seq_len, lstm_size)

        #self.W(hs_bar) (batch, seq_len, lstm_size), ht (batch, lstm_size, 1)
        score = torch.bmm(self.W(hs_bar), ht) # (batch, seq_len, 1)
        return score
        
    def forward(self, hs_bar, ht):
        '''
        Args: 
            hs_bar: all hidden states of the encoder
                of shape (seq_len, batch_size, lstm_size)
            ht: hidden state of decoder at current time t
                of shape (batch_size, lstm_size)
        Returns:
            context_vector: shape (batch_size, lstm_size)
        '''
        score = self.score(hs_bar, ht) # (batch, seq_len, 1)
        
        attn_w = F.softmax(score, dim=1) # (batch, seq_len, 1)
        hs_bar = torch.transpose(hs_bar,0,1) # (batch, seq_len, lstm_size)
        
        context_vector = attn_w * hs_bar # (batch, seq_len, lstm_size)
        context_vector = torch.sum(context_vector, dim=1) # (batch, lstm_size)
        return context_vector


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()  
        self.unroll_len = 30 # For how many steps unroll the lstm

        self.lstm = nn.LSTMCell(args.lstm_size, args.lstm_size)
        self.bn = nn.BatchNorm1d(self.unroll_len * args.lstm_size)
        self.drop = nn.Dropout()
        self.fc = nn.Linear(self.unroll_len * args.lstm_size, args.target_size)

    def forward(self, attention, enc_h, hS):
        '''
        Arguments:
            attn: attention model
            enc_h: all hidden states of the encoder
                of shape (seq_len, batch_size, lstm_size)
            hS: hidden state of the encoder at last time step S
                of shape (1, batch_size, lstm_size)   
        Returns:
            x: predictions by the decoder
                of shape (batch_size, out_size)
        '''
        ht = torch.squeeze(hS, dim=0) # (batch_size, lstm_size)
        ct = torch.zeros_like(ht) # (batch_size, lstm_size)
        dec_h = [] 

        for _ in range(self.unroll_len):
            context_vector = attention(enc_h, ht)
            (ht, ct) = self.lstm(context_vector, (ht, ct))
            # ht (batch_size, lstm_size), ct (batch_size, lstm_size)
            dec_h.append(ht)
        
        x = torch.stack(dec_h, dim=1)  # (batch_size, unroll_len, lstm_size)
        # (batch_size, unroll_len * lstm_size)
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2]) 
        x = self.bn(x) # same shape
        x = self.drop(x) # same shape
        x = F.relu(x) # same shape
        x = self.fc(x) # (batch_size, target_size)
        return x