from itertools import combinations
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_squared_error
import time


def correlation_plot(true, prediction, y_err=None, save_fig=None, dpi=None):
    """
    Graphs the correlation between true values and predicted values and saves the image.
    :param true: list like
        The true values.
    :param prediction: list like
        The predicted values.
    :param y_err: list like
        The errors of the prediction
    :param save_fig: str
        The name of the file to save the image (must include file extension type).
    :param dpi: int
        Specifies the quality of the saved image.
    :return: None
    """
    rmse = math.sqrt(mean_squared_error(true, prediction))
    print(f'Root mean squared error: {rmse}')

    plt.xlabel('Target', fontsize=14)
    plt.ylabel('Prediction', fontsize=14)
    if y_err is not None:
        plt.errorbar(true, prediction, yerr=y_err, c='b', ecolor='red', fmt='o', markersize='1', zorder=1)
    else:
        plt.scatter(true, prediction, c='b', s=1, zorder=1)
    plt.grid()
    if save_fig:
        plt.savefig(save_fig, dpi=dpi)

    return rmse


def kernel_matrices(order, dim, length_scale):
    """
    Helper function used to create the RBF kernels and the matrix input for HDMR.
    :param order: int
        The order of HDMR to use.
    :param dim: int
        The dimension of the feature space.
    :param length_scale: int
        The length scale to use for the HDMR RBF kernels.
    :return:
        1) list of numpy arrays.
            List of matrices, used to select the component functions. Has size (dim choose order).
        2) list of sklearn.guassian_process.kernels.RBF
            List of kernels to use for training. Has size (dim choose order).

    """
    kernels = []
    matrices = []
    matrix = np.eye(dim, dim)
    for c in combinations(range(dim), order):
        matrices.append(matrix[:, list(c)].copy())
        kernels.append(RBF(length_scale=length_scale))
    return matrices, kernels


class RSHDMRGPR:
    def __init__(self, num_models, matrices, kernels):
        """
        Initializes an instance of GPRHDMR object.
        :param num_models: int
            number of Guassian models to be used.
        :param matrices: list of 2D numpy.Arrays
            Represents the linear transformation applied to the data input (No bias term has yet been added).
            The list size must equal num_models.
        :param kernels: a list of Objects from sklearn.guassian_process.kernels
            the kernels to be used for training each HDMR.
        """
        # number of models cannot be empty
        if len(matrices) == 0:
            raise RuntimeError('Must provide atleast one component function.')
        if num_models != len(matrices):
            raise RuntimeError(f'Number of affine transformations must be equal to number of models which is '
                               f'{self.__num_models}')

        num_rows = matrices[0].shape[0]
        for mat in matrices:
            if mat.shape[0] != num_rows:
                raise RuntimeError(f'The rows of the matrices are not all the same.')

        self.__num_models = num_models
        self.__matrices = matrices
        self.__kernels = kernels

        self.__is_trained = False
        self.__models = None

    def train(self, data, label='out', alphas=1e-7, cycles=50, scale_down=(0.5, 1)):
        """
        Trains the 1st order HDMR.
        :param data: DataFrame
            Contains both the the features and the label column
        :param label: str
            The string identifying the label column in data.
        :param alphas: int or list of int
            The noise level to set for each hdmr component function.
        :param cycles: int
            The number of cycles to train. Must be a positive integer.
        :param scale_down: tuple
            A length 2 tuple containing the starting scale factor and the step size.
        :return:
            list of training labels
                list of labels used to carry out all intermediate trainings.
            list of models
                list of the trained models in order.
        """
        # Checks if the model has been trained already. Only trains once.
        if self.__is_trained:
            raise RuntimeError('Model has already been trained')
        if label not in data.columns:
            raise RuntimeError(f'Label column name {label} not found in data.columns.')

        # Validates the noise parameter
        if isinstance(alphas, list):
            if len(alphas) != self.__num_models:
                raise RuntimeError(f'The length of alphas must match {self.__num_models} but received {len(alphas)}')
        if isinstance(alphas, (int, float)):
            alphas = [alphas] * self.__num_models

        # Validates the scale_down argument
        if scale_down is None:
            start, step = 1, 0
        elif isinstance(scale_down, tuple):
            if len(scale_down) != 2:
                raise RuntimeError(f'scale_down must contain 2 elements but received {len(scale_down)}')
            start, step = scale_down[0], scale_down[1]
        else:
            raise RuntimeError(f'scale_down must be of type {tuple} but received {type(scale_down)}')

        # Separates features and labels
        x_train = data.drop(columns=[label])
        y_train = data[label]

        # Initializes the list of models to be trained.
        self.__models = [None] * self.__num_models
        # Initializes the component outputs used for training.
        y_i = [y_train / self.__num_models] * self.__num_models

        for c in range(cycles):
            print(f'Training iteration for cycle {c + 1} has started.')

            for i in range(self.__num_models):
                print('Training component function:', i + 1)
                df = np.dot(x_train, self.__matrices[i])
                gpr = GaussianProcessRegressor(kernel=self.__kernels[i], optimizer=None, n_restarts_optimizer=3,
                                               alpha=alphas[i], random_state=43, normalize_y=False)

                out_i = y_train - sum(y_i) + y_i[i]
                gpr.fit(df, out_i)

                self.__models[i] = gpr
                y_i[i] = gpr.predict(df) * min(start + (1 - start) * step * (c + 1) / cycles, 1)

        self.__is_trained = True
        print('Training completed.')
        return self

    def predict(self, test_data, return_std=False):
        """
        Predicts the label using features from test_data
        :param test_data: DataFrame
            Must have the same number of input features as the data the model was trained on.
        :param return_std: bool
            Returns the standard deviation of the predictions if True.
        :return: numpy.ndarray return_std=False, (np.ndarray, np.ndarray) if return_std=True
            1) The predicted probabilities.
            2) The predicted probabilities and error.
        """
        if not self.__is_trained:
            raise RuntimeError('This model has not been trained. Train the model before telling it to predict.')

        if return_std:
            y_predict = 0
            y_err_sq = 0
            for i in range(self.__num_models):
                df = np.dot(test_data, self.__matrices[i])
                y_pred, y_err = self.__models[i].predict(df, return_std=True)
                y_predict += y_pred
                y_err_sq += y_err * y_err
            return y_predict, np.sqrt(y_err_sq)
        else:
            y_predict = 0
            for i in range(self.__num_models):
                df = np.dot(test_data, self.__matrices[i])
                y_predict += self.__models[i].predict(df)
            return y_predict

    def get_models(self):
        """
        Returns the trained models
        :return: The trained models
        """
        if not self.__is_trained:
            raise RuntimeError("Model must be trained first before it can be returned.")

        return self.__models


class FirstOrderHDMRImpute:
    def __init__(self, models, division=1000):
        """
        Initializes the class by vectorizing the 1D HDMR component functions and creating a dictionary of output values
        with [division] subdivisions.
        :param models: list of trained sklearn.gaussian_process.GaussianProcessRegressor models.
            Contains the hdmr component functions of first order. Must be of first order (1 input and 1 output).
        :param division: int
            The number of divisions in the lookup table
        """
        self.__models = models
        self.__num_models = len(self.__models)
        self.__table_yi = pd.DataFrame(np.linspace(0, 1, division + 1), columns=['input'])

        # Computing the lookup table for HDMR values
        def model_func(x, idx):
            if pd.isna(x):
                return np.nan
            return self.__models[idx].predict(np.array([[x]]))[0]

        self.__model_func = np.vectorize(model_func)

        # table_yi has column order: input, y_0, y_1, ... , y_n
        for i in range(len(models)):
            self.__table_yi['y_' + str(i)] = self.__model_func(self.__table_yi['input'], i)

    def get_yi(self, df_na):
        """
        Modifies the DataFrame to contain the outputs of the first hdmr component functions. If
        :param df_na: pandas DataFrame
            The dataframe to impute. Must contain the 'output' column and must be the last column.
        :return: list
            Indices of the rows corresponding to nan columns.
        """

        n = df_na.shape[1] - 1
        for i in range(self.__num_models):
            df_na['y_' + str(i)] = self.__model_func(df_na.iloc[:, i], i)

        nan_rows = df_na.loc[df_na.isnull().any(axis=1)]
        nan_rows_idx = nan_rows.index
        null_rows = nan_rows.isnull().idxmax(axis=1)

        col_to_ord = {}
        for i in range(n):
            col_to_ord[df_na.columns[i]] = i

        for i in range(null_rows.shape[0]):
            idx = null_rows.index[i]
            col_no = col_to_ord[null_rows.iloc[i]]
            val = df_na['out'].loc[idx] - df_na.loc[null_rows.index[i], :][n + 1:].sum(axis=0)
            df_na.loc[idx, :][n + 1 + col_no] = val

        return nan_rows_idx

    def impute(self, df_na):
        """
        This function imputes the missing values given the input.
        :param df_na: pandas DataFrame
            The DataFrame to impute, should contain the columns corresponding to 1D hdmr outputs.
        :return: pandas DataFrame
            The imputed DataFrame.
        """
        start = time.time()
        nan_rows = df_na.loc[df_na.isnull().any(axis=1)]
        null_entry = nan_rows.isnull().idxmax(axis=1)

        col_to_ord = {}
        for i in range(df_na.shape[1]):
            col_to_ord[df_na.columns[i]] = i

        def get_inputs(num, col, df=self.__table_yi):
            """
            Determines all inputs that correspond to an interval containing num.
            :param num: int
                The number to find the input value for.
            :param col: int
                The column corresponding to the lookup table.
            :param df: pandas DataFrame
                The lookup table.
            :return: list
                List of possible candidates.
            """
            min_yi = df.iloc[:, col].min()
            min_yi_idx = df[df.iloc[:, col] == min_yi]['input'].min()
            max_yi = df.iloc[:, col].max()
            max_yi_idx = df[df.iloc[:, col] == max_yi]['input'].max()

            ret_idxs = []
            for i in range(df.shape[0] - 1):
                left = df.iloc[i, col]
                right = df.iloc[i + 1, col]

                if left <= num < right:
                    ret_idxs.append(df.iloc[i, 0])
                elif right <= num < left:
                    ret_idxs.append(df.iloc[i + 1, 0])

            if len(ret_idxs) == 0:
                if min_yi >= num:
                    ret_idxs.append(min_yi_idx)
                elif max_yi <= num:
                    ret_idxs.append(max_yi_idx)

            return ret_idxs

        for i in range(nan_rows.shape[0]):
            col_no = col_to_ord[null_entry.iloc[i]]
            val = get_inputs(df_na.iloc[i, col_no + self.__num_models + 1], col_no + 1)
            df_na.iloc[i, col_no] = val[0]
        print(f'Function execution time took {time.time() - start} seconds.')
        return df_na
