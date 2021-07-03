import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from linear_model import LinearRegression, LogisticRegression
from termcolor import colored

# R^2 value for machine learning model
def r2_score(y_true, y_predicted):
    corr_matrix = np.corrcoef(y_true, y_predicted)
    corr = corr_matrix[0, 1]
    return corr ** 2

# mean squarred error
def mse(y_true, y_predicted):
    return np.mean((y_true - y_predicted) ** 2)

# accuracy
def accuracy(y_true, y_predicted):
    accuracy = np.sum(y_true == y_predicted) / len(y_true)
    

if __name__ == '__main__':
    
    # linear regression
    heading = colored("Linear-Regression", 'magenta')
    print(colored(f"**************************{heading}**************************", 'blue'))
    
    X_data, target = datasets.make_regression(
        n_samples=100,
        n_features = 1,
        noise = 20,
        )
    X_train, X_test, y_train, y_test = train_test_split(X_data, target, test_size = 0.25)
    
    model = LinearRegression(lr = 0.01, n_iters=1000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    r2 = r2_score(y_test, predictions)
   
    print(colored(f"R^2 values : {colored(r2, 'green', attrs=['bold'])}", 'cyan'))
    print()
    
    # Logistic Regression
    heading = colored("Logistic-Regression", 'magenta')
    print(colored(f"**************************{heading}**************************", 'blue'))
    
    data = datasets.load_breast_cancer()
    X_data, target = data['data'], data['target']
    X_train, X_test, y_train, y_test = train_test_split(X_data, target, test_size=0.25)
    
    model = LogisticRegression(lr = 0.0001, n_iters=1000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    r2 = r2_score(y_test, predictions)
    accu = accuracy(y_test, predictions)
    test_size, correct_class_size = X_test.shape[1], np.sum(y_test == predictions)
    
    print(colored(f"Out of {colored(test_size, 'green', attrs=['bold'])}, {colored(correct_class_size, 'green', attrs=['bold'])} are classified correctly", 'cyan'))
    print(colored(f"Ratio: {colored(accu, 'green')}", 'cyan'))
    print(colored(f"R^2 values : {colored(r2, 'green', attrs=['bold'])}", 'cyan'))    
    
    # TODO : for learning rate 0.001 or higher overflow is encountered for np.exp() 