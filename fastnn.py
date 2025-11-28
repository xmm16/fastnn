#!/usr/bin/python
import sys
import torch
import torch.nn as nn
import torch.optim as optim

# defaults
using_file = False
model = "mlp"
sep = ","
window_size = 3
epochs = 2000

try:
    # updating with user options
    for i in range(len(sys.argv)):
        match sys.argv[i].lower():
            case "--file" | "-f":
                file = sys.argv[i + 1]
                using_file = True

            case "--separator" | "-sp":
                sep = sys.argv[i + 1]

            case "--model" | "-m":
                model = sys.argv[i + 1]

            case "--series" | "-s":
                data = sys.argv[i + 1]
            
            case "--window-size" | "-ws":
                window_size = sys.argv[i + 1]
            
            case "--epochs" | "-e":
                epochs = sys.argv[i + 1]
            
            case "--help" | "-h":
                print("""FastNN: A tool to quickly use prediction networks
To input data:
    --file: select file to parse for data
    OR
    --series: provide series of data as next parameter
Other commands (optional):
    --separator: choose symbol that each datapoint is separated with
    --model: either "MLP", "LSTM", "CNN", or "Transformer"
    --window-size: set window size of model
    --epochs: set epoch value of model
    --help: pull up this menu""")
                sys.exit()

    if using_file:
        with open(file, "r") as f:
            data = f.read()
    
    seq = list(map(float, data.split(sep)))

except Exception as e:
    print("Error parsing parameters")
    yn_error = input("Print error (y/N)? ")

    if yn_error == "y":
        print(e)

    sys.exit()

if len(seq) <= window_size:
    print("Sequence is shorter than window size")
    sys.exit()

def mlp(seq, window_size, epochs):
    X = []
    y = []
    for i in range(len(seq) - window_size):
        X.append(seq[i:i+window_size])
        y.append(seq[i+window_size])
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    model = nn.Sequential(
        nn.Linear(window_size, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )

    loss_fn = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=0.01)

    for _ in range(epochs):
        opt.zero_grad()
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()

    test_input = torch.tensor([seq[-window_size:]], dtype=torch.float32)
    pred = model(test_input).item()
    return pred

def lstm(seq, window_size, epochs):
    X = []
    y = []
    for i in range(len(seq) - window_size):
        X.append(seq[i:i+window_size])
        y.append(seq[i+window_size])
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1) 
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    class LSTMNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
            self.fc = nn.Linear(32, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]) 

    model = LSTMNet()
    loss_fn = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=0.01)

    for _ in range(epochs):
        opt.zero_grad()
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()

    test_input = torch.tensor([seq[-window_size:]], dtype=torch.float32).unsqueeze(-1)
    pred = model(test_input).item()
    return pred

def cnn(seq, window_size, epochs):
    X = []
    y = []
    for i in range(len(seq) - window_size):
        X.append(seq[i:i+window_size])
        y.append(seq[i+window_size])
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    class CNNNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=16, kernel_size=2),
                nn.ReLU(),
                nn.Conv1d(16, 32, kernel_size=2),
                nn.ReLU()
            )
            self.fc = nn.Linear(32 * (window_size - 2), 1)

        def forward(self, x):
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    model = CNNNet()
    loss_fn = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=0.01)

    for _ in range(epochs):
        opt.zero_grad()
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()

    test_input = torch.tensor([seq[-window_size:]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) 
    pred = model(test_input).item()
    return pred

def transformer(seq, window_size, epochs):
    X, y = [], []
    for i in range(len(seq) - window_size):
        X.append(seq[i:i+window_size])
        y.append(seq[i+window_size])
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1) 
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    class TransNet(nn.Module):
        def __init__(self, d_model=16, nhead=2):
            super().__init__()
            self.input_fc = nn.Linear(1, d_model)
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.fc_out = nn.Linear(d_model, 1)

        def forward(self, x):
            x = self.input_fc(x) 
            x = self.transformer(x)  
            return self.fc_out(x[:, -1])  

    model = TransNet()
    loss_fn = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=0.01)

    for _ in range(epochs):
        opt.zero_grad()
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()

    test_input = torch.tensor([seq[-window_size:]], dtype=torch.float32).unsqueeze(-1) 
    pred = model(test_input).item()
    return pred
    
if __name__ == "__main__":
    try:
        match model.lower():
            case "mlp":
                print(mlp(seq, int(window_size), int(epochs)))
    
            case "lstm":
                print(lstm(seq, int(window_size), int(epochs)))
    
            case "cnn":
                print(cnn(seq, int(window_size), int(epochs)))
    
            case "transformer":
                print(transformer(seq, int(window_size), int(epochs)))
    except Exception as e:
        print("Error running model")
        yn_error = input("Print error (y/N)? ")
    
        if yn_error == "y":
            print(e)
    
        sys.exit()
