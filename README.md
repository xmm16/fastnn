# fastnn
Quickly access a prediction neural network model
# Install
- on arch: ```yay -S fastnn```
- on others: ```git clone https://github.com/eshnd/fastnn && cd fastnn && chmod +x fastnn.py && sudo mv fastnn.py /usr/bin/fastnn && cd .. && rm -rf fastnn```
# Why?
- Coming up with numbers that need to follow a pattern
- Needing to analyze a large set of numbers without building an entire script
- Generating a random number similar to other random numbers
# To use
To input data:    
-     --file: select file to parse for data    
OR    
-     --series: provide series of data as next parameter    
Other commands (optional):    
-     --separator: choose symbol that each datapoint is separated with    
-     --model: either "MLP", "LSTM", "CNN", or "Transformer"      
-     --window-size: set window size of model    
-     --epochs: set epoch value of model    
-     --help: pull up this menu    
