# lstm-attention-weather-prediction
- Weather prediction using LSTM and Global Attention. 
- Multivariate time series analysis; the dataset contains 14 different features such as atmospheric pressure, humidity, temperature, etc. The goal is to predict the future temperature using all these 14 features. 

## Dataset
Steps to use the dataset.
- Click this [link](https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip) to download the dataset.
- Extract the zip file. 
- Place the contents inside the extracted folder into the main repository. 

## Instructions
After preparing the dataset, train the neural network:
```
python main.py
```

## Results
<!--- Explain a little bit about the graphs here. -->
- Some sampled predictions from the test set after the model was fully trained. 
- Here, the x-axis is the time in days, and the y-axis denotes the temperature in degC.  

<p align="center">
  <img src="https://user-images.githubusercontent.com/51147727/155930935-be65f7ab-20a7-4caa-92b6-518e8a0df5c9.png"/>
  <img src="https://user-images.githubusercontent.com/51147727/155930943-23e7638f-b550-423e-887c-9cf7b09fa62d.png"/>
</p>
