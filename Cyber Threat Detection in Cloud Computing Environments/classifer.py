from Confusion_mat import multi_confu_matrix
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

def decision_tree(X_train, y_train, X_test, y_test):
    # Scale the features (optional, but can improve performance)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create the Decision Tree model
    model = DecisionTreeClassifier(random_state=42)  # You can adjust hyperparameters as needed

    # Train the model
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_predict = model.predict(X_test_scaled)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_predict)

    # Create confusion matrix
    conf_matrix = confusion_matrix(y_test, y_predict)

    print(f'Test Accuracy: {accuracy}')
    return y_predict, conf_matrix

def logistic_regression(X_train, y_train, X_test, y_test):
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create the Logistic Regression model
    model = LogisticRegression(max_iter=1000)  # You can adjust max_iter as needed

    # Train the model
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_predict = model.predict(X_test_scaled)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_predict)

    # Create confusion matrix
    conf_matrix = confusion_matrix(y_test, y_predict)

    print(f'Test Accuracy: {accuracy}')
    return y_predict, conf_matrix

def SVM(X_train, y_train, X_test, y_test):
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create the SVM model
    model = svm.SVC(kernel='linear')  # You can change the kernel as needed (e.g., 'rbf', 'poly', etc.)

    # Train the model
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_predict = model.predict(X_test_scaled)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_predict)

    # Create confusion matrix
    conf_matrix = confusion_matrix(y_test, y_predict)

    print(f'Test Accuracy: {accuracy}')
    return y_predict, conf_matrix


def xgboost(X_train, y_train, X_test, y_test):
    # Scale the features (optional, but can improve performance)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
    dtest = xgb.DMatrix(X_test_scaled)

    # Set the parameters for XGBoost
    params = {
        'objective': 'multi:softmax',  # For multi-class classification
        'num_class': 6,  # Adjust based on the number of classes
        'max_depth': 3,  # Depth of the tree
        'eta': 0.1,  # Learning rate
        'eval_metric': 'mlogloss',  # Evaluation metric
        'seed': 42  # Random seed for reproducibility
    }

    # Train the model
    model = xgb.train(params, dtrain, num_boost_round=100)

    # Make predictions
    y_predict = model.predict(dtest)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_predict)

    # Create confusion matrix
    conf_matrix = confusion_matrix(y_test, y_predict)

    print(f'Test Accuracy: {accuracy}')
    return y_predict, conf_matrix



def RDNE(X_train, y_train,X_test,y_test):
    # Create individual models
    random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
    decision_tree = DecisionTreeClassifier(random_state=42)

    # Create optimized KNN model (with hyperparameter tuning if necessary)
    knn = KNeighborsClassifier(n_neighbors=5)  # Example with 5 neighbors

    # Combine models into a Voting Classifier
    ensemble_model = VotingClassifier(
        estimators=[
            ('rf', random_forest),
            ('dt', decision_tree),
            ('knn', knn)
        ],
        voting='soft'  # Soft voting uses predicted probabilities
    )

    # Train the ensemble model
    ensemble_model.fit(X_train, y_train)
    # Make predictions
    y_pred = ensemble_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Test Accuracy: {accuracy}')
    cm=multi_confu_matrix(y_test, y_pred)

    return y_pred,cm

