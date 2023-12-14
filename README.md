# Power-Outages-Prediction
This is a project for DSC80 at UCSD
## Framing the Problem
#### Prediction problem: Predict the severity of a major power outage in terms of its duration
This is a regression problem to predict the outage duration of a power outages.

#### Response variable: 'OUTAGE.DURATION'⏱️
The model will try to predict 'OUTAGE.DURATION', a continuous numerical variable that describes how long a power outage lasted (in minutes). This is chosen as the reponse variable because it directly reflects the outage impact's extent as outages that last longer are more "severe".

#### Evaluation Metric: RMSE
Valid evaluation metrics for regression models include RMSE and R^2. The RMSE assesses how well a regression model predicts the value of the response variable in absolute terms while R^2 does so in percentage terms. Because we want to evaluate the model's ability to generalize to unseen data, RMSE is chosen over R^2 as the evaluation metric as it gives a better assessment in how well the model will perform for unseen observations. The lower the RMSE, the better the predictions.

#### Data Cleaning: 
The same data cleaning performed in project 3 will be performed here (and thus the following data cleaning code is from project 3)
- Fix Formatting
    - It appears that due to formatting issues: 
    - the first 4 rows are all NaN values 
    - the correct column names appear in row 5
    - These rows will be dropped and colum names will be reassigned.
- Typecasting
    - All columns are stored as strings, but it would make more sense for numerical information to be stored as floats.
    - It would be preferable if the power outages start date and time were merged into one pd.Timestamp column, and power outages restoration date and time were merged into one pd.Timestamp column.
- Fill in missing values
  - The response variable 'OUTAGE.DURATION' has 58 NaN values, so median imputation will be performed because its distribution is skewed.
- Keeping relevant columns
    - The DataFrame now has 57 columns. Since we're concerned with predicting the severity of a power outage from other power outage features, we only need columns in the dataset that include information related outage severity and other potentially contextual variables. We also only want information we would know at the time of prediction, so no columns that contain information that could only be known after a power outage will be included.
    - The chosen relevant columns are:
      - YEAR: Indicates the year when the outage event occurred
      - MONTH: Indicates the month when the outage event occurred
      - U.S._STATE: Represents all the states in the continental U.S.
      - POSTAL.CODE: Represents the postal code of the U.S. states
      - NERC.REGION: The North American Electric Reliability Corporation (NERC) regions involved in the outage event
      - CLIMATE.REGION: U.S. Climate regions as specified by National Centers for Environmental Information (nine climatically consistent regions in continental U.S.A.)
      - CLIMATE.CATEGORY: This represents the climate episodes corresponding to the years. The categories—“Warm”, “Cold” or “Normal” episodes of the climate are based on a threshold of ± 0.5 °C for the Oceanic Niño Index (ONI)
      - OUTAGE.START.DATE: This variable indicates the day of the year when the outage event started (as reported by the corresponding Utility in the region)
      - OUTAGE.START.TIME: This variable indicates the time of the day when the outage event started (as reported by the corresponding Utility in the region)
      - OUTAGE.RESTORATION.DATE: This variable indicates the day of the year when power was restored to all the customers (as reported by the corresponding Utility in the region)
      - OUTAGE.RESTORATION.TIME: This variable indicates the time of the day when power was restored to all the customers (as reported by the corresponding Utility in the region)
      - CAUSE.CATEGORY: Categories of all the events causing the major power outages
      - CAUSE.CATEGORY.DETAIL: Detailed description of the event categories causing the major power outages
      - OUTAGE.DURATION: Duration of outage events (in minutes)
      - PI.UTIL.OFUSA: State utility sector׳s income (earnings) as a percentage of the total earnings of the U.S. utility sector׳s income (in %)
      - TOTAL.CUSTOMERS: Annual number of total customers served in the U.S. state

The first 5 rows of the resulting DataFrame:
| U.S._STATE | POSTAL.CODE | NERC.REGION | CLIMATE.REGION | OUTAGE.START         | OUTAGE.RESTORATION    | OUTAGE.DURATION | YEAR | MONTH | CAUSE.CATEGORY | CAUSE.CATEGORY.DETAIL | CLIMATE.CATEGORY | PI.UTIL.OFUSA | TOTAL.CUSTOMERS |
|------------|-------------|-------------|-----------------|----------------------|-----------------------|------------------|------|-------|----------------|-----------------------|-------------------|---------------|------------------|
| Minnesota  | MN          | MRO         | East North Central | 2011-07-01 17:00:00 | 2011-07-03 20:00:00 | 3060.0           | 2011 | 7.0   | severe weather | NaN                   | normal            | 2.2           | 2595696          |
| Minnesota  | MN          | MRO         | East North Central | 2014-05-11 18:38:00 | 2014-05-11 18:39:00 | 1.0              | 2014 | 5.0   | intentional attack | vandalism           | normal            | 2.2           | 2640737          |
| Minnesota  | MN          | MRO         | East North Central | 2010-10-26 20:00:00 | 2010-10-28 22:00:00 | 3000.0           | 2010 | 10.0  | severe weather | heavy wind            | cold              | 2.1           | 2586905          |
| Minnesota  | MN          | MRO         | East North Central | 2012-06-19 04:30:00 | 2012-06-20 23:00:00 | 2550.0           | 2012 | 6.0   | severe weather | thunderstorm         | normal            | 2.2           | 2606813          |
| Minnesota  | MN          | MRO         | East North Central | 2015-07-18 02:00:00 | 2015-07-19 07:00:00 | 1740.0           | 2015 | 7.0   | severe weather | NaN                   | warm              | 2.2           | 2673531          |

Now that the data is cleaned, a baseline model can be built.

## Baseline Model
#### Model
The model used in this prediciton task will be a **RandomForestRegressor**, a model suitable for regression and relatively robust to overfitting, therefore performing better on unseen data.
#### Features
In previous exploration of the dataset, it was found that certain regions have longer power outages. Thus, it seems that locational factors could be related to outage duration. The selected features for the model are:
- **'CLIMATE.REGION'** 🗺️
  > U.S. Climate regions as specified by National Centers for Environmental Information (nine climatically consistent regions in continental U.S.A.)
  
  > This is a nominal categorical variable

- **'NERC.REGION'** 📍
  > The North American Electric Reliability Corporation (NERC) regions involved in the outage event
  
  > This is a nominal categorical variable

These columns provide relevant locational information and are accessible at the time of prediction.
#### Feature Engineering
Since 'CLIMATE.REGION' and 'NERC.REGION' are both nominal categorical variables, they will be converted into numerical representations so that are suitable to predict 'OUTAGE.DURATION'.
- **'CLIMATE.REGION'** 🗺️: One-Hot Encoding

One-hot encoding will transform the single categorical feature into multiple numerical features, creating binary columns for each unique category that indicate the presence or absence of that category in the data. The potential categories in 'CLIMATE.REGION' are: 'East North Central', 'Central', 'South', 'Southeast', 'Northwest', Southwest', 'Northeast', 'West North Central', and 'West'.
explanation 

- **'NERC.REGION'** 📍: One-Hot Encoding
  
The potential categories in 'NERC.REGION' are: 'MRO', 'SERC', 'RFC', 'ECAR', 'TRE', 'WECC', 'SPP', 'NPCC', 'FRCC', 'FRCC, SERC'
After one-hot encoding the 2 nominal categorical features, the baseline RandomForestRegressor model will use 19 discrete numerical features:
> CLIMATE.REGION (East North Central)🗺️

> CLIMATE.REGION (Central)🗺️

> CLIMATE.REGION (South)🗺️

> CLIMATE.REGION (Southeast)🗺️

> CLIMATE.REGION (Northwest)🗺️

> CLIMATE.REGION (Southwest)🗺️

> CLIMATE.REGION (Northeast)🗺️

> CLIMATE.REGION (West North Central)🗺️

> CLIMATE.REGION (West)🗺️

> NERC.REGION (MRO)📍

> NERC.REGION (SERC)📍

> NERC.REGION (RFC)📍

> NERC.REGION (ECAR)📍

> NERC.REGION (TRE)📍

> NERC.REGION (WECC)📍

> NERC.REGION (SPP)📍

> NERC.REGION (FRCC)📍

> NERC.REGION (NPCC)📍

> NERC.REGION (FRCC, SERC)📍


**Note**: The model will actually use 17 features, because for each original feature ('CLIMATE.REGION' and 'NERC.REGION'), the model will drop one one-hot encoded feature in order to prevent multicollinearity. Additionally, 'CLIMATE.REGION' has 6 missing values, a trivial portion of the dataset, so those rows will be dropped.
