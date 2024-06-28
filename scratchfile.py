
""" 
np.int = np.int32
np.float = np.float64
np.bool = np.bool_

 """

import pickle

def main():

    # Load the best_parameters structure from disk
    with open('best_parameters.pkl', 'rb') as f:
        best_parameters = pickle.load(f)

    # Load the best_results structure from disk
    with open('best_results.pkl', 'rb') as f:
        best_results = pickle.load(f)

    print("BEST PARAMETERS: \n")
    print(best_parameters)

    print('BEST RESULTS: \n')
    print(best_results)

if __name__ == "__main__":
    main()