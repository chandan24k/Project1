import numpy as np

class ElasticNetModel:
    def __init__(self, alpha, l1_ratio, max_iter, tol=1e-6, learning_rate=0.01):
        """
        Initialize the ElasticNet regression model.

        Parameters:
        alpha: Regularization strength (λ)
        l1_ratio: The mixing ratio between L1 and L2 (0 <= l1_ratio <= 1)
        max_iter: Maximum number of iterations for gradient descent
        tol: Tolerance for stopping criterion
        learning_rate: Step size for gradient descent
        """
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate

    def _soft_threshold(self, rho, l1_penalty):
        """Soft thresholding operator for L1 penalty."""
        if rho < -l1_penalty:
            return rho + l1_penalty
        elif rho > l1_penalty:
            return rho - l1_penalty
        else:
            return 0

    def _compute_loss(self, X, y, coef_, intercept_):
        """Compute the ElasticNet loss (MSE + L1 + L2 penalties)."""
        y_pred = X.dot(coef_) + intercept_
        mse_loss = np.mean((y - y_pred) ** 2)
        l1_penalty = self.l1_ratio * np.sum(np.abs(coef_))
        l2_penalty = (1 - self.l1_ratio) * np.sum(coef_ ** 2)
        return mse_loss + self.alpha * (l1_penalty + l2_penalty)

    def fit(self, X, y):
        """
        Fit the model to the data using gradient descent.

        Parameters:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        """
        n_samples, n_features = X.shape

        # Normalize the features
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X = (X - X_mean) / X_std

        # Initialize weights and intercept
        coef_ = np.zeros(n_features)
        intercept_ = 0
        loss_history = []

        for iteration in range(self.max_iter):
            y_pred = X.dot(coef_) + intercept_
            residuals = y_pred - y

            # Compute gradient for intercept
            intercept_grad = np.sum(residuals) / n_samples
            intercept_ -= self.learning_rate * intercept_grad

            # Compute gradient for coefficients (ElasticNet penalty)
            coef_grad = X.T.dot(residuals) / n_samples + \
                        self.alpha * (self.l1_ratio * np.sign(coef_) +
                                      (1 - self.l1_ratio) * 2 * coef_)

            # Update weights
            coef_ -= self.learning_rate * coef_grad

            # Record the loss
            loss = self._compute_loss(X, y, coef_, intercept_)
            loss_history.append(loss)

            # Stopping condition (based on gradient tolerance)
            if np.linalg.norm(coef_grad) < self.tol:
                break

        # Return the fitted model and results encapsulated in ElasticNetModelResults
        return ElasticNetModelResults(coef_, intercept_, X_mean, X_std, loss_history)

class ElasticNetModelResults:
    def __init__(self, coef_, intercept_, X_mean, X_std, loss_history):
        """
        Encapsulates the results of the ElasticNet model after fitting.

        Parameters:
        coef_: Fitted coefficients for the model
        intercept_: Fitted intercept for the model
        X_mean: Mean of the features (used for normalization)
        X_std: Standard deviation of the features (used for normalization)
        loss_history: History of the loss values during training
        """
        self.coef_ = coef_
        self.intercept_ = intercept_
        self.X_mean = X_mean
        self.X_std = X_std
        self.loss_history = loss_history

    def predict(self, X):
        """
        Predict target values for given input features.

        Parameters:
        X: Feature matrix for which predictions are to be made

        Returns:
        y_pred: Predicted target values
        """
        # Normalize the input data with the same scaling applied in fit
        X = (X - self.X_mean) / self.X_std
        return X.dot(self.coef_) + self.intercept_

    def plot_loss_history(self):
        """
        Plot the history of the loss function during training.
        """
        import matplotlib.pyplot as plt
        plt.plot(self.loss_history)
        plt.title("Loss History")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.show()

    def print_summary(self):
        """
        Print a summary of the fitted model, including coefficients and intercept.
        """
        print("Model Summary:")
        print(f"Intercept: {self.intercept_}")
        print(f"Coefficients: {self.coef_}")
        print(f"Number of iterations: {len(self.loss_history)}")
        print(f"Final loss: {self.loss_history[-1]}" if self.loss_history else "No loss recorded.")
