# Read X and Y .npy files and process them to create a dataset for training
# and testing the model. The dataset is saved as a .data file.

import numpy as np


def process_data(X, Y, output_file):
    # Get the number of samples
    num_samples = X.shape[0]
    num_past_windows = X.shape[1]
    # Get the number of features
    num_features = X.shape[2]
    # Get the number of classes
    num_classes = Y.shape[1]
    num_output = Y.shape[0]
    
    print('Number of samples:', num_samples)
    print('Number of features:', num_features)
    print('Number of classes:', num_classes)
    print('Number of output:', num_output)

    # Create the dataset
    with open(output_file, 'w') as f:
        # Write the number of samples, features and classes
        f.write(str(num_samples) + ' ' + str(num_past_windows * num_features) + ' ' + str(num_classes) + '\n')
        # Write the samples
        for i in range(num_samples):
            # Get the sample
            sample = X[i]
            
            for j in range(num_past_windows):
                for k in range(num_features):
                    if j == num_past_windows - 1 and k == num_features - 1:
                        f.write(str(sample[j][k]) + '\n')
                    else:
                        f.write(str(sample[j][k]) + ' ')
                    
            for j in range(num_classes):
                if j == num_classes - 1:
                    f.write(str(Y[i][j]) + '\n')
                f.write(str(Y[i][j]) + ' ')

def evaluate_results(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        count = 0
        sum_error = 0
        for line in lines:
            data = line.split(',')
            prediction = np.array([float(x) for x in data[0].split(' ')])
            real = np.array([float(x) for x in data[1].split(' ')])
            
            error = np.abs(prediction - real)
            sum_error += np.sum(error)
            count += 2
    
    print('Mean error:', sum_error / count)

            
def main():
    #valuate_results('test_results.txt')
    #return
    
    X_train = np.load('X_train.npy')
    Y_train = np.load('Y_train.npy')
    X_test = np.load('X_test.npy')
    Y_test = np.load('Y_test.npy')
    
    
    process_data(X_train, Y_train, 'train.data')
    process_data(X_test, Y_test, 'test.data')
    
    
if __name__ == '__main__':
    main()