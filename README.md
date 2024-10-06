# Project 1 
Group Members:-

* Chandan Kumar - CWID - A20525237 
* Nanda Kishore Thummala - CWID - Assds
* Niharika


Linear Regression with ElasticNet Regularization

Overview
This implementation provides a Linear Regression model with ElasticNet regularization, which combines L1 (Lasso) and L2 (Ridge) regularization. This model is useful for scenarios where you need both feature selection and coefficient shrinkage to prevent overfitting.

Usage
1.Initialization: Create an instance of ElasticNetRegression with desired parameters.
2.Training: Use the fit method to train the model on your data.
3.Prediction: Use the predict method to make predictions on new data.

Example
# Example usage
model = ElasticNetRegression(alpha=0.1, rho=0.5, iterations=1000, learning_rate=0.01)
model.fit(X, y)
predictions = model.predict(X)
print("Coefficients:", model.theta)

Questions
1. What does the model you have implemented do and when should it be used?

The model performs linear regression with ElasticNet regularization, which is useful when you need to balance between feature selection (L1 regularization) and coefficient shrinkage (L2 regularization). It is particularly effective for high-dimensional data.

2. How did you test your model to determine if it is working reasonably correctly?

The model was tested using synthetic data where the true relationship between features and the target variable is known. The coefficients learned by the model were compared to the true coefficients to ensure accuracy.

3. What parameters have you exposed to users of your implementation in order to tune performance?

alpha: Overall regularization strength.

rho: Balance between L1 and L2 regularization.

iterations: Number of iterations for gradient descent.

learning_rate: Step size for gradient descent.

4. Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental to the model?

The implementation may struggle with very large datasets or datasets with highly correlated features. Given more time, techniques such as feature scaling, normalization, and advanced optimization algorithms could be implemented to improve performance.

