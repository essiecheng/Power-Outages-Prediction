# Power-Outages-Prediction
This is a project for DSC80 at UCSD
### Framing the Problem
#### Prediction problem: Predict the severity of a major power outage in terms of its duration
This is a regression problem to predict the outage duration of a power outages.

#### Response variable: 'OUTAGE.DURATION'⏱️
The model will try to predict 'OUTAGE.DURATION', a continuous numerical variable that describes how long a power outage lasted (in minutes). This is chosen as the reponse variable because it directly reflects the outage impact's extent as outages that last longer are more "severe".

#### Evaluation Metric: RMSE
Valid evaluation metrics for regression models include RMSE and R^2. The RMSE assesses how well a regression model predicts the value of the response variable in absolute terms while R^2 does so in percentage terms. Because we want to evaluate the model's ability to generalize to unseen data, RMSE is chosen over R^2 as the evaluation metric as it gives a better assessment in how well the model will perform for unseen observations. The lower the RMSE, the better the predictions.
