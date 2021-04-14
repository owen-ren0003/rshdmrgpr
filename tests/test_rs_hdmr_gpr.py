import io
import unittest
from unittest.mock import patch

from sklearn.model_selection import train_test_split

from rshdmrgpr.rs_hdmr_gpr import *


class TestRSHDMRGPR(unittest.TestCase):
    def setUp(self):
        def f0(*args):
            sumed = 0
            for a in [*args]:
                sumed += a

            return sumed

        # Creates a synthetic data set for testing
        column_names = []
        n = 3
        for i in range(n):
            column_names.append(f'a{i}')
        np.random.seed(42)  # Fixes the random seed so the results are always the same.
        data = pd.DataFrame(np.random.rand(1000, n), columns=column_names)
        inputs = [data.iloc[:, i] for i in range(n)]
        data['out'] = f0(*inputs)
        scale = data['out'].max() - data['out'].min()
        trans = data['out'].min()
        data['out'] = (data['out'] - trans) / scale
        # Creates a training and testing set
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(data.drop(columns=['out']), data['out'], train_size=100, test_size=None, random_state=42)

        matrices, kernels = kernel_matrices(1, n, length_scale=0.6)
        # Initializes
        self.model = RSHDMRGPR(matrices, kernels)

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_v_print(self, mock_stdout):
        """
        Tests the verbose_print member method
        """
        # Tests if output is printed correctly when on=True
        RSHDMRGPR.verbose_print("Something", on=True)
        self.assertEqual(mock_stdout.getvalue(), "Something\n")

        # Clears the io.StringIO object for next test
        mock_stdout.seek(0)
        mock_stdout.truncate(0)

        # Tests if the output
        RSHDMRGPR.verbose_print("something", end='', on=True)
        self.assertEqual(mock_stdout.getvalue(), "something")

        mock_stdout.seek(0)
        mock_stdout.truncate(0)

        RSHDMRGPR.verbose_print("something", end='', on=False)
        self.assertEqual(mock_stdout.getvalue(), "")

    def test_train(self):

        # Test exception raised by incorrect data shape
        with self.assertRaises(RuntimeError) as context:
            self.model.train(self.x_train.iloc[:, :-1], self.y_train)
        self.assertEqual(context.exception.args[0], "Number of columns in provided data which is 2 does not match "
                                                    "number of rows which is 3 in each linear transformation")

        # Test invalid alpha arguments
        with self.assertRaises(RuntimeError) as context:
            self.model.train(self.x_train, self.y_train, alphas=[1e-5, 1e-7])
        self.assertEqual(context.exception.args[0], "The length of alphas must match 3 but received 2")
        with self.assertRaises(RuntimeError) as context:
            self.model.train(self.x_train, self.y_train, alphas=[1e-5, 1e-7, 's'])
        self.assertEqual(context.exception.args[0], "A non-float value s was provided as the noise")
        with self.assertRaises(RuntimeError) as context:
            self.model.train(self.x_train, self.y_train, alphas='4')
        self.assertEqual(context.exception.args[0], "Provided alpha is not a float or list of floats")

        # Test invalid scale_down argument
        with self.assertRaises(RuntimeError) as context:
            self.model.train(self.x_train, self.y_train, scale_down=(1, 1, 1))
        self.assertEqual(context.exception.args[0], "scale_down must contain 2 elements but received 3")
        with self.assertRaises(RuntimeError) as context:
            self.model.train(self.x_train, self.y_train, scale_down=[1, 1])
        self.assertEqual(context.exception.args[0], f"scale_down must be of type {tuple} but received {list}")

        # Test invalid use_columns argument
        with self.assertRaises(RuntimeError) as context:
            self.model.train(self.x_train, self.y_train, use_columns=[True, True, 's'])
        self.assertEqual(context.exception.args[0], "use_columns must be a list of bool of length 3.")
        with self.assertRaises(RuntimeError) as context:
            self.model.train(self.x_train, self.y_train, use_columns=[True, True, False, True])
        self.assertEqual(context.exception.args[0], "use_columns must be a list of bool of length 3.")

        # Test invalid verbose argument
        with self.assertRaises(RuntimeError) as context:
            self.model.train(self.x_train, self.y_train, verbose=3)
        self.assertEqual(context.exception.args[0], "The valid levels of verbose are: 0, 1, or 2, please choose one.")


class TestHelpers(unittest.TestCase):
    def test_load_data(self):
        """
        Tests the load_data helper function
        """
        data1 = load_data('h2o')
        self.assertEqual(data1.shape, (10001, 4))

        data2 = load_data('KED')
        self.assertEqual(data2.shape, (585890, 8))

        data3 = load_data('financial')
        self.assertEqual(data3.shape, (3927, 15))

    def test_kernel_matrices(self):
        """
        Tests the kernel_matrices helper function
        """
        matrices, kernels = kernel_matrices(3, 5, kernel_function=RBF, length_scale=0.6)

        self.assertEqual(len(matrices), len(kernels))
        self.assertEqual(len(matrices), 10)
        for i in range(len(matrices)):
            self.assertEqual(matrices[i].shape, (5, 3))
            self.assertEqual(kernels[i].length_scale, 0.6)

        with self.assertRaises(RuntimeError) as context:
            kernel_matrices(6, 5, kernel_function=RBF, length_scale=0.6)
        self.assertEqual(context.exception.args[0], "order must be larger than 1 and less than dim which is 5 but 6 "
                                                    "was given.")


if __name__ == '__main__':
    unittest.main()
