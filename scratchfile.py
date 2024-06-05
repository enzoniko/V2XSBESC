
""" 
np.int = np.int32
np.float = np.float64
np.bool = np.bool_

 """

from preprocessing import preprocess
from models import BaselineModels


def main():
    # Load the dataframes
    X_train_df, X_test_df, Y_train_df, Y_test_df = preprocess()

    # Train and evaluate the baseline models
    baseline_models = BaselineModels()

    baseline_models.train_and_evaluate(X_train_df, X_test_df, Y_train_df, Y_test_df)


if __name__ == "__main__":
    main()