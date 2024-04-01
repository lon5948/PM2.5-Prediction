import argparse
import pandas as pd
import numpy as np

def ReadTrainData(path):
    df = pd.read_csv(path, encoding='big5')
    df = df.iloc[:, 3:] # Remove Location, Date, ItemName
    df = df.applymap(lambda x: x.strip() if type(x) == str else x) # Remove trailing whitespaces
    df.replace(['#', '*', 'x', 'A'], '0.0', inplace=True)
    train_data = df.values
    train_data = np.reshape(train_data, (12, -1, 18, 24)) # shape: (12, 20, 18, 24)
    train_data = train_data.swapaxes(1, 2).reshape(12, 18, -1) # shape: (12, 18, 480)
    train_data = train_data.astype(float)
    return train_data

def ReadTestData(path):
    df = pd.read_csv(path, header=None, encoding='big5')
    df = df.iloc[:, 2:] # Remove Location, Date, ItemName
    df = df.applymap(lambda x: x.strip() if type(x) == str else x) # Remove trailing whitespaces
    df.replace(['#', '*', 'x', 'A'], '0.0', inplace=True)
    test_data = df.values
    test_data = test_data.reshape(-1, 18, 9) # shape: (244, 18, 9)
    test_data = test_data.astype(float)
    return test_data

def extract_feature(data, selected, square, cube):
    x_data = []
    y_data = []
    for month in range(data.shape[0]):
        for i in range(data.shape[2] - 9):
            A = data[month, selected, i:i+9].flatten()
            B = data[month, square, i:i+9].flatten()
            C = data[month, cube, i:i+9].flatten()
            # R = np.multiply(data[month, 9, i:i+9], data[month, 7, i:i+9])
            Z = np.concatenate((A, B**2, C**3), axis=0)
            x_data.append(Z)
            y_data.append(data[month, 9, i+9])
    return np.array(x_data), np.array(y_data)

def normalization(x_data):
    mean = np.mean(x_data, axis=0)
    std = np.std(x_data, axis=0)
    x_data = (x_data - mean) / std
    return mean, std, x_data

def train(x_data, y_data, length_of_features):
    b = 0.0
    w = np.ones(length_of_features * 9)
    lr = 0.5
    epoch = 50000
    b_lr = 0.0
    w_lr = np.zeros(length_of_features  * 9)
    lambda_value = 0
    
    for e in range(epoch):
        # y_data = b + w * x_data
        error = y_data - b - np.dot(x_data, w) 

        # Calculate gradient
        b_grad = -2 * np.sum(error) * 1
        w_grad = -2 * np.dot(error, x_data) + 2 * lambda_value * w
        
        # Update sum of squares of gradients
        b_lr = b_lr + np.square(b_grad)
        w_lr = w_lr + np.square(w_grad)

        # Update parameters
        b = b - lr / np.sqrt(b_lr) * b_grad
        w = w - lr / np.sqrt(w_lr) * w_grad
        
        loss = np.mean(np.square(error)) + lambda_value * np.sum(np.square(w))
        
        if (e + 1) % 1000 == 0:
            print(f'epoch {e + 1}: Loss {np.sqrt(loss)}')
    return b, w
        
def main(args):
    # Train
    train_data = ReadTrainData(args.train)
    selected_features = [2, 3, 5, 6, 8, 9, 13]
    square_features = [2, 3, 8, 9]
    cube_features = [8, 9]
    length_of_features = len(selected_features) + len(square_features) + len(cube_features)
    x_data, y_data = extract_feature(train_data, selected_features, square_features, cube_features)
    mean, std, x_data = normalization(x_data)
    b, w = train(x_data, y_data, length_of_features)
    
    # Test
    X_test = ReadTestData(args.test)
    
    with open(args.output, 'w+') as f:
        f.write('index,answer\n')
        for i in range(X_test.shape[0]):
            A = X_test[i, selected_features, :].flatten()
            B = X_test[i, square_features, :].flatten()
            C = X_test[i, cube_features, :].flatten()
            Z = np.concatenate((A, B**2, C**3), axis=0)
            Z = (Z - mean) / std
            f.write('index_{},{}\n'.format(i, b + np.dot(w, Z)))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HW1: Linear Regression")
    parser.add_argument("--train", type=str, default='train.csv', help="path of train file (default: 'train.csv')")
    parser.add_argument("--test", type=str, default='test.csv', help="path of test file (default: 'test.csv')")
    parser.add_argument("--output", type=str, default='output.csv', help="path of output file (default: 'output.csv')")
    main(parser.parse_args())
    
    