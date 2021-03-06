import math
from itertools import combinations
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_squared_error


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
        plt.tight_layout()
        plt.savefig(save_fig, dpi=dpi)
    return rmse


def kernel_matrices(order, dim, length_scale, length_scale_bounds=(1e-05, 100000.0)):
    """
    Helper function used to create the RBF kernels and the matrix input for HDMR.

    :param order: int
        The order of HDMR to use.
    :param dim: int
        The dimension of the feature space.
    :param length_scale: int
        The length scale to use for the HDMR RBF kernels.
    :param length_scale_bounds: tuple
        The length scale bounds to use.
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
        kernels.append(RBF(length_scale=length_scale, length_scale_bounds=length_scale_bounds))
    return matrices, kernels


class RSHDMRGPR:
    def __init__(self, num_models, matrices, kernels):
        """
        Initializes an instance of RS-HDMR-GPR object.

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
        self.__columns = [True] * self.__num_models

    def train(self, x_train, y_train, alphas=1e-7, scale_down=(0.2, 2), cycles=50, optimizer=None, opt_every=5,
              use_columns=None, n_restarts=3, initializer='even', report_rmse=False):
        """
        Trains the component functions.

        :param x_train: pandas DataFrame
            Contains the features.
        :param y_train: list or 1d-array
            The label column of the data.
        :param alphas: float or list of float
            The noise level to set for each hdmr component function. Should be small.
        :param scale_down: tuple
            A length 2 tuple containing the starting scale factor and the step size. start represents the starting
            fraction of scale down, and step size represents a rate at which this fraction increases.
        :param cycles: int
            The number of cycles to train. Must be a positive integer.
        :param optimizer: str or list of str
            Must be a GPR optimizer or list of such. Please see sklearn documentation for GaussianProcessRegressor
            optimizer.
        :param opt_every: int
            Specifies every (opt_every) cycles to apply the optimizer. Default=1 (optimizer applied every cycle). This
            control variable does nothing if the optimizer argument is not provided
        :param use_columns: list of bool or None
            Specifies which column to use (True indicates use, False otherwise). If None (not specified), all columns
            are used.
        :param n_restarts: int
            Positive integer indicating the number of restarts on training. Does nothing meaningful if the optimizer is
            not provided.
        :param initializer: list of float or 'even'
            Initializes the starting targets for each component function. If 'even', every target is initialized as
            y_train / (Number of True in use_columns).
        :param report_rmse: bool
            Saves the rmse from the prediction on the training set over cycles.
        :return: Returns 1) if report_rmse = True, 2) otherwise.
            1) self, pandas DataFrame
                The first argument is the trained instance of self. The second contains the RMSE of predicted
                values (vs actual) on the training set for each cycle.
            2) self
                The trained instance of self.
        """

        # Checks if the model has been trained already. Should only train once.
        if self.__is_trained:
            raise RuntimeError('Model has already been trained')

        # Validates the noise parameter
        if isinstance(alphas, list):
            if len(alphas) != self.__num_models:  # Checks   that number of alphas is equal to the number of models
                raise RuntimeError(f'The length of alphas must match {self.__num_models} but received {len(alphas)}')
            for a in alphas:  # Every alpha provided must be an int or float
                if not isinstance(a, (int, float)):
                    raise RuntimeError(f'A non-float value {a} was provided as the noise.')
        if isinstance(alphas, (int, float)):  # if just a float or int is provided, all noise will be set to this noise
            alphas = [alphas] * self.__num_models

        # Decides which component function to use in training
        if use_columns is not None:
            if len(use_columns) == self.__num_models and all(isinstance(a, bool) for a in use_columns):
                self.__columns = use_columns
            else:
                raise RuntimeError(f"use_columns must be a list of bool of length {self.__num_models}.")

        # Validates the scale_down argument
        if scale_down is None:
            start, step = 1, 0
        elif isinstance(scale_down, tuple):
            if len(scale_down) != 2:
                raise RuntimeError(f'scale_down must contain 2 elements but received {len(scale_down)}')
            start, step = scale_down[0], scale_down[1]
        else:
            raise RuntimeError(f'scale_down must be of type {tuple} but received {type(scale_down)}')

        # Validates the optimizer parameter
        if isinstance(optimizer, list):
            if len(optimizer) != self.__num_models:
                raise RuntimeError(f'optimizer provided as a list must have length equal to the number of'
                                   f'{self.__num_models} but received {len(alphas)}.')
        elif isinstance(optimizer, str):
            optimizer = [optimizer] * self.__num_models
        elif optimizer is None:
            optimizer = [optimizer] * self.__num_models

        # Initializes the list of models to be trained.
        self.__models = [None] * self.__num_models

        # Initializes the component outputs used for training.
        if initializer == 'even':
            y_i = [y_train / sum(self.__columns)] * self.__num_models
            for i in range(self.__num_models):
                if not self.__columns[i]:
                    y_i[i] = 0
        elif isinstance(initializer, list):
            y_i = [initializer[i] * y_train for i in range(self.__num_models)]

        # These variables are only used if report_rmse is True
        rmse_val = []
        cycle_no = []

        self.__is_trained = True
        for c in range(cycles):
            print(f'Training iteration for CYCLE {c + 1} has started.')

            for i in range(self.__num_models):

                # Decides which cycle to use optimizer (last cycle will always use)
                if c % opt_every == 0 or c == cycles - 1:
                    opt = optimizer[i]
                else:
                    opt = None
                if c != 0:
                    # Sets the length_scale to the previous round of length_scales
                    self.__kernels[i] = self.__models[i].kernel_

                if self.__columns[i]:
                    print(f'Training component function: {i + 1}, Optimizer: {opt},', end=" ")

                    df = np.dot(x_train, self.__matrices[i])
                    gpr = GaussianProcessRegressor(kernel=self.__kernels[i], optimizer=opt,
                                                   n_restarts_optimizer=n_restarts, alpha=alphas[i], random_state=43,
                                                   normalize_y=False)

                    # subtracts every element from y_i except for the i-th variable
                    out_i = y_train - sum(y_i) + y_i[i]
                    gpr.fit(df, out_i)

                    print(f'Resulting length_scale: {gpr.kernel_}')
                    self.__models[i] = gpr  # replaces the gpr model from the previous cycle
                    y_i[i] = gpr.predict(df) * min(start + (1 - start) * step * (c + 1) / cycles, 1)
                else:
                    print(f'Component {i + 1} was omitted from training.')

            if report_rmse:
                # Calculates and record the RMSE on the training set for each cycle
                predicted = self.predict(x_train)
                rmse_val.append(math.sqrt(mean_squared_error(predicted, y_train)))
                cycle_no.append(c + 1)

        print('Training completed.')
        if report_rmse:
            return self, pd.DataFrame({'cycle_no': cycle_no, 'rmse': rmse_val})
        return self

    def predict(self, test_data, return_std=False):
        """
        Predicts the label using features from test_data.

        :param test_data: DataFrame
            Must have the same number of input features as the data the model was trained on.
        :param return_std: bool
            Returns the standard deviation of the predictions if True.
        :return: 1) return_std=False, 2) otherwise.
            1) numpy.ndarray, numpy.ndarray
                The predicted values and the square-root of the sum of variances.
            2) numpy.ndarray
                The predicted values
        """
        if not self.__is_trained:
            raise RuntimeError('This model has not been trained. Train the model before telling it to predict.')

        if return_std:
            y_predict = 0
            y_err_sq = 0
            for i in range(self.__num_models):
                if self.__columns[i]:
                    df = np.dot(test_data, self.__matrices[i])
                    y_pred, y_err = self.__models[i].predict(df, return_std=True)
                    y_predict += y_pred
                    y_err_sq += y_err * y_err
            return y_predict, np.sqrt(y_err_sq)
        else:
            y_predict = 0
            for i in range(self.__num_models):
                if self.__columns[i]:
                    df = np.dot(test_data, self.__matrices[i])
                    y_predict += self.__models[i].predict(df)
            return y_predict

    def get_models(self):
        """
        Returns the trained component function models.

        :return: list of GaussianProcessRegressor
            The trained hdmr component functions
        """
        if not self.__is_trained:
            raise RuntimeError("Model must be trained first before it can be returned.")

        return self.__models


def sequential_fitting(x_train, y_train, models, **params):
    """
    This function fits a sequence of RSHDMRGPR models sequentially.

    :param x_train:
    :param y_train:
    :param models: list of RSHDMRGPR models
    :param params: dict
        variable number of key word arguments
    :return: list of numpy arrays

    """
    default = {
        'alphas': 1e-6,  # can be different for each model
        'cycles': 50,
        'scale_down': (0.2, 2),
        'optimizer': 'fmin_l_bfgs_b',
        'opt_every': 5,
        'use_columns': None,
        'n_restarts': 1,
        'initializer': 'even'
    }

    for key in params:
        if key not in default:
            raise RuntimeError(f'{key} is not a hyperparameter.')
        default[key] = params[key]

    y = [y_train]
    for i in range(len(models)):
        print(f'\nMODEL {i} with {len(models)} component functions has started training.')
        models[i].train(x_train, y[i], alphas=default['alphas'][i], cycles=default['cycles'],
                        scale_down=default['scale_down'], optimizer=default['optimizer'],
                        opt_every=default['opt_every'], use_columns=default['use_columns'],
                        n_restarts=default['n_restarts'], initializer=default['initializer'])
        y_predict = models[i].predict(x_train)
        y.append(y[i] - y_predict)

    return y


def batch_predict(model, data, batch_size=2000, report_size=50000):
    y_predf = []
    i = 0
    while i < data.shape[0]:
        y_pred = model.predict(data.iloc[i: i + batch_size])
        y_predf.extend(y_pred)
        i += batch_size
        if i % report_size == 0:
            print(f'{i} batches have been predicted.')
    return np.array(y_predf)


class FirstOrderHDMRImpute:
    def __init__(self, models, division=1000):
        """
        Initializes the class by vectorizing the 1D HDMR component functions and creating a lookup DataFrame whose first
        column are the endpoints of [division] subdivisions of the interval [0, 1] and the remaining columns are the
        y_i := f_i(column_1) where f_i is the i-th HDMR component function.

        :param models: list of GaussianProcessRegressor
            Contains the trained HDMR component functions of first order. Must be of first order (1 input and 1 output).
        :param division: int
            The number of divisions in the lookup table.
        """
        self.__models = models
        self.__num_models = len(self.__models)
        self.__table_yi = pd.DataFrame(np.linspace(0, 1, division + 1), columns=['input'])
        self.__gap = 2 / division  # Used for imputation to set minimum distance threshold.

        # Computes the lookup table for the first component function values
        def model_func(x, idx):
            if pd.isna(x):
                return np.nan
            return self.__models[idx].predict(np.array([[x]]))[0]

        self.__model_func = np.vectorize(model_func)

        # table_yi has column order: input, y_0, y_1, ... , y_n
        for i in range(len(models)):
            self.__table_yi['y_' + str(i)] = self.__model_func(self.__table_yi['input'], i)

    def get_table(self):
        """
        Returns the lookup table for the component functions.

        :return: panda DataFrame
            Returns the lookup table for the HDMR component functions.
        """
        return self.__table_yi

    def get_yi(self, df_na):
        """
        Modifies the DataFrame to contain the outputs of the first HDMR component functions.

        Adds n additional columns to df_na, where n is the number of feature columns in df_na. The label column
        must be the last column or the function won't work.

        :param df_na: pandas DataFrame
            The dataframe to impute. Must contain the 'output' column and must be the last column.
        :return: list
            Indices of the rows corresponding to nan columns.
        """

        n = df_na.shape[1] - 1

        # Creates the extra columns to the right of the last column of df_na that contains the outputs of the features
        # from the trained component functions. For instance y_0 will be the output of the 0th indexed column of df_na.
        for i in range(self.__num_models):
            df_na['y_' + str(i)] = self.__model_func(df_na.iloc[:, i], i)

        nan_rows = df_na.loc[df_na.isnull().any(axis=1)]  # Extracts the rows with missing values.
        nan_rows_idx = nan_rows.index  # Extracts the index of those rows with missing values
        nan_cols = nan_rows.isnull().idxmax(axis=1)  # Selects the corresponding column name to the missing value entry

        # column to index dictionary for lookup
        col_to_idx = {}
        for i in range(n):
            col_to_idx[df_na.columns[i]] = i

        # Computes y_i corresponding to the missing value column.
        for i in range(nan_cols.shape[0]):
            idx = nan_rows_idx[i]  # index of the ith null row occurrence
            col_no = col_to_idx[nan_cols.iloc[i]]  # the column number
            val = df_na.loc[idx, :][n] - df_na.loc[idx, :][n + 1:].sum(axis=0)
            df_na.loc[idx, :][n + 1 + col_no] = val

        return nan_rows_idx

    def impute(self, df_na, get_candidates=False, threshold=0.001):
        """
        This function imputes the missing values given the input.

        :param df_na: pandas DataFrame
            The DataFrame to impute, should contain the columns corresponding to 1D hdmr outputs (i.e. have get_yi
            called upon it).
        :param get_candidates: bool
            Option to return the imputed candidates or not.
        :param threshold: float
            Set the threshold distance for selecting from look-up table.
        :return: 1) if get_candidates=True, 2) otherwise
            1) (pandas DataFrame, pandas Index, pandas Series, list of float)
                The imputed df_na, the index of rows with null entries, column name (indexed by the index of null
                entries) containing the null entry, and list of candidates for imputing that missing value.
            2) pandas DataFrame
                The imputed df_na.
        """
        start = time.time()
        # Selects the rows with missing values.
        nan_rows = df_na.loc[df_na.isnull().any(axis=1)].copy()
        # Selects the corresponding column name to the missing value entry.
        null_entry = nan_rows.isnull().idxmax(axis=1)

        # column name to column index dictionary.
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
            :return: list of float
                List of possible candidates.
            """
            # Obtains the candidates within self.__gap distance
            min_yi = df.iloc[:, col].min()
            min_yi_idx = df.loc[abs(df.iloc[:, col] - min_yi) <= self.__gap, 'input'].tolist()
            max_yi = df.iloc[:, col].max()
            max_yi_idx = df.loc[abs(df.iloc[:, col] - max_yi) <= self.__gap, 'input'].tolist()

            ret_inputs = []
            for i in range(df.shape[0]):
                val = df.iloc[i, col]
                if i < df.shape[0] - 1:
                    left = df.iloc[i, col]
                    right = df.iloc[i + 1, col]
                    if left <= num <= right:
                        ret_inputs.append(df.iloc[i, 0])
                    elif right <= num < left:
                        ret_inputs.append(df.iloc[i + 1, 0])
                    elif abs(val - num) < threshold:
                        ret_inputs.append(df.iloc[i, 0])
                elif abs(val - num) < threshold:
                    ret_inputs.append(df.iloc[i, 0])

            if min_yi >= num:
                ret_inputs.extend(min_yi_idx)
            elif max_yi <= num:
                ret_inputs.extend(max_yi_idx)

            return ret_inputs

        # Will return all the possible imputation candidates if this option is set to True.
        if get_candidates:
            candidates = []

        for i in range(nan_rows.shape[0]):
            col_no = col_to_ord[null_entry.iloc[i]]  # changes the column name to column index.
            val = get_inputs(nan_rows.iloc[i, col_no + self.__num_models + 1], col_no + 1)
            nan_rows.iloc[i, col_no] = val[0]

            if get_candidates:
                candidates.append(val)

        df_na.loc[df_na.isnull().any(axis=1)] = nan_rows
        print(f'Function execution time took {time.time() - start} seconds.')

        if get_candidates:
            return df_na, nan_rows.index, null_entry, candidates
        return df_na
