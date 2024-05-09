## Task
Implement **linear regression** using only numpy to predict the value of PM2.5
| [kaggle link](https://www.kaggle.com/t/4241d949d7dd49ac8a9b518994347166)

## Dataset 
Hsinchu meteorological observation data form Central Weather Bureau.

- #### [Training Data](https://drive.google.com/file/d/1Ly8FfrUSgOTNA3xhsbaJaLS3KYsc3OpJ/view?usp=sharing)
    Climate data for the first 20 days of each month.

- #### [Testing Data](https://drive.google.com/file/d/1nVts8Hcx4iFplVeNRYyEdsUaGfMxGNdE/view?usp=sharing)
    Sample continuous data for 10 hours from the remaining 10 days of each month. Use data from the first 9 hours as features and PM 2.5 from the last hour as the target.

## Implementation

#### Preprocessing

While reading the raw data, originally, a process was required for every 18 lines read. To facilitate access, I stored the raw data in a 2D array of size 18 X 5760, where each row represents various different features, and each column represents data for each hour. Additionally, since some fields contain the symbols ‘#’, ‘*’, ‘x’, ‘A’, these values need to be converted to the floating-point number ‘0.0’.

#### Feature Selection

Calculate the correlation coefficient between each feature and the PM2.5 feature, and filter out the features that have a higher impact on PM2.5.

In this assignment, I discarded all features with a correlation coefficient less than 0.3. Additionally, for features with a correlation coefficient greater than 0.5, a new feature of the square of the original feature will be added, and for those with a correlation coefficient greater than 0.8, a new feature of the cube of the original feature will also be added.

## Run

#### Calculate the Correlation Coefficient
```bash!
python correlation.py {path_of_train_data}
```

#### Prediction
```bash!
python main.py --train {path_of_train_data} --test {path_of_test_data} --output {path_of_result}
```
