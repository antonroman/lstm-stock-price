# lstm-stock-price-predictor

## Introduction
```lstm-stock-price-predictor``` is a Python3 script using Keras running on top of Tensorflow to predict stock values.

Initial version completely based on this tutorial: https://stackabuse.com/time-series-analysis-with-lstm-using-pythons-keras-library/

This script receives as input two csv files and the name of the stock. The usage is as follows:

```bash
lstm_stock_value_predictor.py -t <trainingfile> -i <testingfile> -n <stock name>
```

- ```-t --training```: name of the CSV file which contains the stock for a number of years (5 years seems to be a right value). This samples will be used to train the model.
- ```-i --testing```: name of the CVS file used to test the model. We are going to predict the values for this month and check how similar the values are by plotting them in a graph. 
- ```-n --name```:  name of the stock to be used in the graphs. 

The result will be a graph comparing the predicted and the actual values.

## How to get the stock CSV files
The stock values were obtained from [Yahoo Finnance portal](https://finance.yahoo.com): .
1. These are the steps you must follow to get the CSV files
2. Go to https://finance.yahoo.com 
3. Search for the symbol of the stock you want to predict.
4. Click on historical data.
5. Select the data range you want to download and click 'Apply'.
6. Download the CSV file by clicking on 'Download data' button.
7. Repeat this process for the training and testing samples.

## Requiremets

The script is written in Python and it uses the following libraries:
- **Numpy**: for basic algebraic operations. 
- **Pandas**: for data manipulation.
- **Keras**: for the LSTM model.
- **Tensorflow**: the machine learning engine used by Keras

### How to install the dependencies:
**OS**: Ubuntu 18.04 LTS

```sudo apt install python3-numpy python3-pandas python3-keras python3-sklearn ```

I installed Tensorflow using ```pip```:

```sudo python3 -m pip install tensorflow```