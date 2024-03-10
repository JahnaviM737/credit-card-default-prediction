# Credit Card Default Prediction

## Project Overview
The goal of the project is to predict credit card default for each customer using anonymized customers' financial data. Based on each customer’s past 13 monthly statements, default occurs when the customer is expected to not pay their due amount in 120 days after their last statement date. There are 189 features belonging to one of these five feature profiles: Delinquency, Spend, Payment, Balance, and Risk. Data is sourced from an AMEX Kaggle competition.

## Data pre-processing
There are 5.53m records for 459k customers. The target variable is provided at the customer-level (positive class: customer has defaulted on their credit card payment). There are 177 numerical features, 11 categorical features, and one time dimension (statement date).

### Data condensation
In managing the colossal size of our 'train_data' dataset (16.39 GBs), we strategically employed chunk-based loading and manipulated column data types. This reduced the dataset to 1.5 GBs, optimizing memory usage. Techniques included converting 'Customer_ID' from 64 bytes to 4 bytes using the last 16 characters in hexadecimal, compressing 'S_2' with pd.to_datetime(), and converting 11 categorical columns from 8 to 1 byte. For 177 numeric columns, we reduced from 8 to 2 bytes using float16, ensuring efficient processing without sacrificing data integrity.

### Handling Missing values
Out of the 189 features in the dataset, 122 of them have at least one missing value. On average each feature has 15% of their values missing. There are 30 features with more that 50% missing values. We have dropped these features. For the rest of the features, we used the feature’s mean to impute the missing values.

### Data Aggregation / Feature Engineering
In addressing the challenge of having multiple statements for each customer while possessing only customer-level target variable, we employ a robust data aggregation strategy. Our goal is to distill the essential information from the individual statements, aggregating the data at the customer level. We have added 5 aggregated functions (min, max, mean, standard deviation, and last) for each numerical feature and 2 aggregated functions (last and mode) for each categorical feature. When we aggregate these features, we are applying each aggregate function to all statements of each customer.

#### Numerical Variables Aggregation:
For each unique customer ID, we perform aggregation on numerical variables by creating five distinct aggregates:
1. Mean: Calculating the average value across all statements.
2. Standard Deviation: Measuring the dispersion of values to capture variability.
3. Min: Identifying the minimum value among all statements.
4. Max: Determining the maximum value across all statements.
5. Last Statement Value: Capturing the value from the latest statement.

This approach significantly enriches the dataset by introducing multiple perspectives on numerical features, enhancing the representation of each customer's financial profile.

#### Categorical Variables Aggregation:
For categorical variables, we adopt a similar approach, generating two meaningful aggregates:
1. Mode: Identifying the most frequently occurring category across all statements.
2. Last Statement Value: Capturing the categorical value from the latest statement.

By applying these aggregation functions to each categorical variable, we enhance the categorical feature set, ensuring a comprehensive representation of each customer's categorical behavior.

The resultant dataset, post-aggregation, expands the feature space by a factor of five for numerical features and two for categorical features. This multiplication is a trade-off for collapsing the approximately 13 rows per customer into a single row. This strategic aggregation not only aligns with our data preprocessing goals but also sets the stage for more efficient customer-level analyses. The aggregated dataset provides a holistic and condensed view, laying the foundation for effective modeling and decision-making at the customer level.

### Standardization and Resampling
The source data is already scaled and normalized. The 10 categorical features were handled by a mix of one-hot encoding and ordinal encoding. In credit card default prediction, resampling is typically necessary to handle the target variable imbalance. In our case, however, the negative class was already undersampled. The negative class has been subsampled for this dataset at 5%. As a result of that the positive class is around 26% (negative class at around 74%). We did not resample the data, but when splitting the data to development and test, we have used stratified sampling to ensure the class distribution remains consistent.

## Modeling
We chose tree-based models as our primary ones because they are capable of handling categorical variables and missing values, and improved upon the baseline models with boosted trees for they focus on misclassified points—which is important to avoid costly false negatives. One model which was significantly different from the rest was the regularized logistic regression model, which we trained to check if the data could be linearly separable.

### Model performance Comparison
We trained baseline versions of all models in order to determine which one would be best suited for this dataset. From this initial exploration we removed logistic regression from our radar for it failed to predict any instances of the positive class, and we also determined that XGBoost would likely be our model of choice. The non-tuned XGBoost model achieved an accuracy of 0.88 and an F-1 score of 0.77—noticeably higher numbers than the untuned versions of HistGradientBoosting, Adaboost, random forest, and decision tree. It also achieved a AUC-ROC score of 0.94 and an average precision score of 0.85, which are promising results.

### Hyperparameter tuning
We dove into the hyperparameter tuning in our XGBoost model, which had already demonstrated the most promising results among all baseline models. The initial step involved a systematic grid search approach, focusing on a select range of key hyperparameters including max_depth, gamma, n_estimators, min_child_weight, subsample, and learning_rate. This grid search was conducted to ensure a thorough exploration within the defined parameter space. Then we conducted a random search around the promising regions identified in the grid search, and refined the model's parameters more accurately. Finally, combined with feature engineering steps, our optimized XGBoost model achieved 88.6% of accuracy, which outperformed the baseline version by 7.9%.

### Precision / Recall tradeoff
In the context of the credit card default prediction, the choice between precision and recall depends on specific financial risk tolerance and the operational cost associated with false predictions. If the cost associated with false positives (model predicts customer will default, but the customer does not default) is high, optimizing for higher precision is critical. If the cost with false negative (missing actual customer default) is high, optimizing recall is more critical. Since we did not have any input with regards to cost associated with false predictions, our modeling efforts focused on overall performance but we wanted to call out the importance of this tradeoff for default prediction.

### Feature Importance
Our top 10 most important features determined by optimized XGBoost model were B_33_last, B_18_last, R_2_last, R_10_mean, R_1_mean, R_2_mean, B_33_mean, B_8_mean, R_10_std, and D_120_last. B_33_last has the highest score of 0.5299 by a long shot, where the next highest is B_18_last with a score of 0.05. Unfortunately with the anonymized dataset provided, we are unable to dive deeper into what each specific feature represents, but we can dissect that the latest Balance data (B_last) and average Risk (R_mean) of each customer has the most impact on our predictions. Also, by analyzing the feature importances, we selected the top 150 important features for model training and tuning, significantly reducing the time cost by 78.9%.
