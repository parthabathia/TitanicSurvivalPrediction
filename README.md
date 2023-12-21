# Titanic Survival Prediction

The provided code performs a logistic regression analysis on the Titanic dataset to predict the survival status of passengers. Here's a breakdown of the code:

1. **Importing Libraries:**
   - NumPy is imported as `np`.
   - Pandas is imported as `pd`.
   - Matplotlib is imported as `plt`.
   - Seaborn is imported as `sns`.
   - Necessary modules from scikit-learn are imported: `train_test_split`, `LogisticRegression`, and `accuracy_score`.

2. **Loading and Exploring Data:**
   - Titanic dataset is loaded from a CSV file (`'titanic.csv'`) using Pandas.
   - The shape, first few rows (`head()`), and information (`info()`) of the dataset are displayed.
   - Null values in the dataset are checked using `isnull().sum()`.

3. **Data Preprocessing:**
   - The 'Cabin' column is dropped from the dataset.
   - Missing values in the 'Age' column are filled with the mean of the column.
   - Missing values in the 'Embarked' column are filled with the mode of the column.

4. **Data Visualization:**
   - Various count plots are created using Seaborn to visualize the distribution of survival status, gender, and passenger class.

5. **Label Encoding:**
   - The 'Sex' and 'Embarked' columns are converted to numerical values using label encoding.

6. **Data Splitting:**
   - The dataset is split into features (X) and the target variable (Y).
   - The data is then split into training and testing sets using `train_test_split`.

7. **Logistic Regression Model:**
   - A logistic regression model is created using `LogisticRegression()` from scikit-learn.
   - The model is trained on the training data.

8. **Model Evaluation:**
   - The accuracy of the model is evaluated on both the training and testing sets.

9. **Testing on New Data:**
   - A new dataset (`'titanic_test.csv'`) is loaded for testing the trained model.
   - Similar data preprocessing steps are applied to this dataset.
   - The trained model is used to predict survival on the new data.

10. **Result Presentation:**
    - The predictions on the new data are organized into a DataFrame.
    - The results are saved to a CSV file (`'titanic_test_result.csv'`).

The overall goal of the code is to train a logistic regression model on the Titanic dataset, validate its performance, and then use the trained model to predict the survival status of passengers in a new dataset. The results are saved in a CSV file for further analysis or submission.
