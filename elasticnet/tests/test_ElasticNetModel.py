import os
import csv
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # This is one of the options, you can also use 'Qt5Agg', etc.
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from elasticnet.models.ElasticNet import ElasticNetModel


def test_predict():
    print("Current Working Directory:", os.getcwd())
    data = []
    csv_path = os.path.join(os.path.dirname(__file__), "small_test.csv")
    with open(csv_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    X = np.array([[float(v) for k,v in datum.items() if k.startswith('x')] for datum in data])
    y = np.array([float(datum['y']) for datum in data])
    # y = np.array([[float(v) for k,v in datum.items() if k=='y'] for datum in data])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = ElasticNetModel(alpha=0.5, l1_ratio=0.5, max_iter=10000, learning_rate=0.01)
    results = model.fit(X_train,y_train)
    preds = results.predict(X_test)
    print("Predicted values:", preds)
    print("Actual values:", y_test)
    print("Differences:", np.abs(preds - y_test))
    results.plot_loss_history()
    results.print_summary()

    # Plot predictions vs actual values
    plt.scatter(y_test, preds, color='blue', label='Predicted')
    plt.plot(y_test, y_test, color='red', label='Actual', linewidth=2)
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.title('Predicted vs Actual')
    plt.legend()
    plt.show()

        # assert preds == 0.5
    tolerance = 10  # Example tolerance
    assert np.all(np.abs(preds - y_test) < tolerance), "Predictions do not match expected values within the tolerance."


    
if __name__ == "__main__":
    test_predict()

   

