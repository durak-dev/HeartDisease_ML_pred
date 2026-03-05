import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

def calculate_F1(X_train, y_train, X_test, y_test, interval):
    """
    Trains RandomForestClassifiers for different max_leaf_nodes values on
    a fixed training set and evaluates on a fixed test set.
    """
    y_train_mapped = y_train.map({'Presence': 1, 'Absence': 0})
    y_test_mapped = y_test.map({'Presence': 1, 'Absence': 0})

    train_results = []
    test_results = []
    for value in interval:
        rf = RandomForestClassifier(max_leaf_nodes=value, random_state=42)
        rf.fit(X_train, y_train_mapped)
        # --- Train Score ---
        train_pred = rf.predict(X_train)
        f1_train = f1_score(y_train_mapped, train_pred)
        train_results.append(round(f1_train, 3))
        # --- Test Score ---
        test_pred = rf.predict(X_test)
        f1_test = f1_score(y_test_mapped, test_pred)
        test_results.append(round(f1_test, 3))

    best_train_index = np.argmax(train_results)
    best_test_index = np.argmax(test_results)

    best_train_value = interval[best_train_index]
    best_test_value = interval[best_test_index]

    selected_positions = interval[::10]

    print(f'The best F1 score for train is {max(train_results)} with {best_train_value} Max Leaf Nodes')
    print(f'The best F1 score for test is {max(test_results)} with  {best_test_value} Max Leaf Nodes')

    plt.figure(figsize=(10, 6))
    plt.plot(interval, train_results, label="Train F1")
    plt.plot(interval, test_results, label="Test F1")

    plt.xticks(selected_positions)
    plt.legend()
    plt.ylabel("F1 score")
    plt.xlabel('Max Leaf Nodes')
    plt.title(f"Random Forest Performance vs Max Leaf Nodes")
    plt.grid(True)
    plt.show()