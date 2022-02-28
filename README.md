# lstm-attention-weather-prediction
Here, we do weather prediction using LSTM and Attention. The dataset contains 14 different features such as atmospheric pressure, humidity, temperature, etc. Our goal is to predict the future temperature using all these 14 features. 

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
Some sampled predictions from the test set after the model is fully trained. Here, the x-axis denotes the time in days, and the y-axis is the temperature in degC.  

<!--- ![img1](https://user-images.githubusercontent.com/51147727/155930935-be65f7ab-20a7-4caa-92b6-518e8a0df5c9.png) -->
<!--- ![img2](https://user-images.githubusercontent.com/51147727/155930943-23e7638f-b550-423e-887c-9cf7b09fa62d.png) -->

<p align="center">
  <img src="https://user-images.githubusercontent.com/51147727/155930935-be65f7ab-20a7-4caa-92b6-518e8a0df5c9.png"/>
  <img src="https://user-images.githubusercontent.com/51147727/155930943-23e7638f-b550-423e-887c-9cf7b09fa62d.png"/>
</p>

<!--- Explain a little bit about the graphs here. -->
