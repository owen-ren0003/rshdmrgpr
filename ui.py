import gc

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from rshdmrgpr.rs_hdmr_gpr1 import *

matplotlib.use("TkAgg")


class DataLoader(tk.Frame):

    data = None
    default = pd.DataFrame({'': ['RS-HDMR-GPR', 'https://github.com/owen-ren0003/rshdmrgpr']})

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.grid_propagate(0)  # prevents frame from resizing

        # font style
        s = ttk.Style()
        s.configure('my.TButton', font=('Helvetica', 15))

        load_button = ttk.Button(self, text='Load Data', width=40, command=self.load_data, style='my.TButton')
        load_button.grid(row=2, column=0, sticky=tk.E + tk.S)
        clear_button = ttk.Button(self, text='Clear Existing Data', width=40, command=self.clear_data,
                                  style='my.TButton')
        clear_button.grid(row=2, column=0, sticky=tk.W + tk.S)
        self.summary_label = None

        self.columnconfigure(0, weight=1)

        #####################################################################################
        # Treeview is ued for displaying the data
        #####################################################################################
        style = ttk.Style(self)
        # set ttk theme to "clam" which support the fieldbackground option
        style.theme_use("clam")
        style.configure("Treeview")
        self.treeview = ttk.Treeview(self)
        self.treeview.grid(row=0, column=0, columnspan=2, sticky=tk.S + tk.W + tk.E + tk.N)
        self.rowconfigure(0, weight=1)

        # Adds the horizontal and vertical scroll bars
        self.treeview_scrollbary = ttk.Scrollbar(self, orient="vertical", command=self.treeview.yview)
        self.treeview_scrollbary.grid(row=0, column=0, columnspan=2, sticky=tk.S + tk.E + tk.N)
        self.treeview_scrollbarx = ttk.Scrollbar(self, orient="horizontal", command=self.treeview.xview)
        self.treeview_scrollbarx.grid(row=1, column=0, columnspan=2, sticky=tk.S + tk.E + tk.W + tk.N)
        self.treeview.configure(yscrollcommand=self.treeview_scrollbary.set,
                                xscrollcommand=self.treeview_scrollbarx.set)

        # Replace with default values
        self.treeview['columns'] = list(DataLoader.default.columns)
        for i in self.treeview['columns']:
            self.treeview.column(i, anchor="w")
            self.treeview.heading(i, text=i, anchor='w')
        for index, row in DataLoader.default.iterrows():
            self.treeview.insert("", 0, text=self.default.shape[0] - 1 - index, values=list(row))
        self.treeview.column('#0', width=100)

    def load_data(self):
        """Loads and displays the data inside Treeview"""
        filename = filedialog.askopenfilename(title="Select A File",
                                              file=(("csv files", "*.csv"),
                                                    ("dat files", "*.dat"),
                                                    ("excel files", "*.xlsx"),
                                                    ("All Files", "*.*")))
        file_path = filename
        try:
            filename = f"{file_path}"
            name = os.path.splitext(os.path.basename(filename))[0]
            if name in ['h2o', 'KED', 'financial']:
                DataLoader.data = load_data(name)
            else:
                DataLoader.data = pd.read_csv(filename)
        except ValueError:
            messagebox.showerror("Information", "The file you have chosen is invalid.")
        except FileNotFoundError:
            messagebox.showerror("Information", f"No such file as {file_path}")
        self.clear_tree()

        self.treeview['columns'] = list(DataLoader.data.columns)
        for i in self.treeview['columns']:
            self.treeview.column(i, anchor="w")
            self.treeview.heading(i, text=i, anchor='w')

        for index, row in DataLoader.data.iterrows():
            self.treeview.insert("", 0, text=self.data.shape[0] - 1 - index, values=list(row))
        self.treeview.column('#0', width=100)

        self.summary_label = ttk.Label(self, text=f'Data shape: {DataLoader.data.shape}', width=40)
        self.summary_label.grid(row=2, column=0, columnspan=2, sticky=tk.S + tk.N)

    def clear_tree(self):
        """Used to clear the loaded data displayed in Treeview"""
        self.treeview.delete(*self.treeview.get_children())

    def clear_data(self):
        """Clears the data from memory"""
        if DataLoader.data is None:
            return

        self.clear_tree()
        # Clears the Header
        self.treeview['columns'] = []
        for i in self.treeview['columns']:
            self.treeview.column(i, anchor="w")
            self.treeview.heading(i, text=i, anchor='w')
        # Clears the Data

        DataLoader.data = None
        gc.collect()
        self.summary_label.destroy()

        # Replace with default values
        self.treeview['columns'] = list(DataLoader.default.columns)
        for i in self.treeview['columns']:
            self.treeview.column(i, anchor="w")
            self.treeview.heading(i, text=i, anchor='w')
        for index, row in DataLoader.default.iterrows():
            self.treeview.insert("", 0, text=self.default.shape[0] - 1 - index, values=list(row))
        self.treeview.column('#1', width=500)


class Trainer(tk.Frame):

    model = None
    y_pred = None
    y_true = None

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.grid_propagate(0)  # prevents frame from resizing

        #####################################################################################
        # Sets the training hyperparameters
        #####################################################################################

        title1 = ttk.Label(self, text='Training Hyperparameters', width=24)
        title1.config(font=("Courier", 14))
        title1.grid(row=0, column=0, columnspan=2, sticky=tk.W)
        self.rowconfigure(0, weight=1)

        row_pos = {
            "alpha": 1,
            "n_restarts": 4,
            "cycles": 7,
            "scale_down": 10,
            "optimizer_false": 13,
            "opt_every": 16
        }
        col_pos = {
            "alpha": 0,
            "n_restarts": 0,
            "cycles": 0,
            "optimizer": 0
        }

        # alpha parameter
        alpha_label = ttk.Label(self, text='alphas (overall noise level):', width=40)
        alpha_label.grid(row=row_pos["alpha"], column=col_pos["alpha"], columnspan=2, sticky="nsew")
        self.alpha_text = ttk.Entry(self, width=40)
        self.alpha_text.insert(0, 1e-7)
        self.alpha_text.grid(row=row_pos["alpha"] + 1, column=0, columnspan=2, sticky=tk.W+ tk.E)

        # n_restarts parameter
        n_restarts_label = ttk.Label(self, text='n_restarts (number of restarts on optimizer):', width=40)
        n_restarts_label.grid(row=row_pos["n_restarts"], column=0, columnspan=2, sticky=tk.W + tk.E)
        self.n_restarts_text = ttk.Entry(self, width=40)
        self.n_restarts_text.insert(0, 1)
        self.n_restarts_text.grid(row=row_pos["n_restarts"] + 1, column=0, columnspan=2, sticky=tk.W + tk.E)

        # cycle parameter
        cycle_label = ttk.Label(self, text='cycles (number of cycles to train):', width=40)
        cycle_label.grid(row=row_pos["cycles"], column=0, columnspan=2, sticky=tk.W + tk.E)
        self.cycle_text = ttk.Entry(self, width=40)
        self.cycle_text.insert(0, 50)
        self.cycle_text.grid(row=row_pos["cycles"] + 1, column=0, columnspan=2, sticky=tk.W + tk.E)

        # scale_down parameter
        scale_down_label = ttk.Label(self, text='scale_down: (starting fraction, step size)', width=40)
        scale_down_label.grid(row=row_pos["scale_down"], column=0, columnspan=2, sticky=tk.W + tk.E)
        self.scale_down1_text = ttk.Entry(self, width=15)
        self.scale_down1_text.insert(0, 0.2)
        self.scale_down1_text.grid(row=row_pos["scale_down"] + 1, column=0, sticky=tk.W + tk.E)
        self.scale_down2_text = ttk.Entry(self, width=15)
        self.scale_down2_text.insert(0, 2)
        self.scale_down2_text.grid(row=row_pos["scale_down"] + 1, column=1, sticky=tk.W + tk.E)

        # optimizer parameter
        optimizer_label = ttk.Label(self, text='optimizer (GPR optimizer to use):', width=40)
        optimizer_label.grid(row=row_pos["optimizer_false"], column=0, columnspan=2, sticky=tk.W + tk.E)
        self.var = tk.IntVar()
        self.optimizer_false_button = ttk.Radiobutton(self, text='None', value=0, variable=self.var)
        self.optimizer_false_button.grid(row=row_pos["optimizer_false"] + 1, column=0, sticky=tk.W + tk.E)
        self.optimizer_true_button = ttk.Radiobutton(self, text='fmin_l_bfgs_b', value=1, variable=self.var)
        self.optimizer_true_button.grid(row=row_pos["optimizer_false"] + 1, column=1, sticky=tk.W + tk.E)

        # opt_every parameter
        opt_every_label = ttk.Label(self, text='opt_every (number of cycles to apply optimizer):', width=40)
        opt_every_label.grid(row=row_pos["opt_every"], column=0, columnspan=2, sticky=tk.W + tk.E)
        self.opt_every_text = ttk.Entry(self, width=40)
        self.opt_every_text.insert(0, 5)
        self.opt_every_text.grid(row=row_pos["opt_every"] + 1, column=0, columnspan=2, sticky=tk.W + tk.E)

        #####################################################################################
        # Set the model hyperparameters
        #####################################################################################

        start_pos = 24
        title = ttk.Label(self, text='Model Hyperparameters', width=21)
        title.config(font=("Courier", 14))
        title.grid(row=start_pos, column=0, columnspan=2, sticky=tk.W + tk.E)
        self.rowconfigure(start_pos, weight=1)

        hdmr_dim_label = ttk.Label(self, text='Dimension of Terms', width=40)
        hdmr_dim_label.grid(row=start_pos + 1, column=0, columnspan=2, sticky=tk.W + tk.E)
        self.hdmr_dim_text = ttk.Entry(self, width=20)
        self.hdmr_dim_text.insert(0, 1)
        self.hdmr_dim_text.grid(row=start_pos + 2, column=0, columnspan=2, sticky=tk.W + tk.E)

        length_scale_label = ttk.Label(self, text='Length Scale', width=40)
        length_scale_label.grid(row=start_pos + 3, column=0, columnspan=2, sticky=tk.W + tk.E)
        self.length_scale_text = ttk.Entry(self, width=20)
        self.length_scale_text.insert(0, 0.6)
        self.length_scale_text.grid(row=start_pos + 4, column=0, columnspan=2, sticky=tk.W + tk.E)

        train_size_label = ttk.Label(self, text='Size of Training Set', width=40)
        train_size_label.grid(row=start_pos + 5, column=0, columnspan=2, sticky=tk.W + tk.E)
        self.train_size_text = ttk.Entry(self, width=20)
        self.train_size_text.insert(0, 500)
        self.train_size_text.grid(row=start_pos + 6, column=0, columnspan=2, sticky=tk.W + tk.E)

        #####################################################################################
        # Button to initiate training
        #####################################################################################

        self.save_mod = ttk.Button(self, text="Save Model", width=40, command=self.save_model)
        self.save_mod.grid(row=start_pos + 8, column=0, columnspan=2, sticky=tk.S)
        self.rowconfigure(start_pos+8, weight=3)

        self.save_pred = ttk.Button(self, text="Save Prediction with Data", width=40, command=self.save_prediction)
        self.save_pred.grid(row=start_pos + 9, column=0, columnspan=2, sticky=tk.S)

        self.load_model_predict = ttk.Button(self, text="Load Model and Predict", width=40,
                                             command=self.load_and_predict)
        self.load_model_predict.grid(row=start_pos + 10, column=0, columnspan=2, sticky=tk.S)

        self.train_button = ttk.Button(self, text="Train New Model", width=40, command=self.train_button_action)
        self.train_button.grid(row=start_pos+11, column=0, columnspan=2, sticky=tk.S)

        self.plot_scatter_button = ttk.Button(self, text="Correlation Plot", width=40, command=self.plot_scatter)
        self.plot_scatter_button.grid(row=start_pos + 12, column=0, columnspan=2, sticky=tk.S)

        self.test_rmse = None
        self.r_squared = None

    def plot_scatter(self):
        """ Plots the correlation plot between the prediction and actual value after model has been trained """
        if Trainer.y_pred is None or Trainer.y_true is None:
            messagebox.showerror("Information", "Please train the model first before plotting")
            return

        fig = plt.figure(figsize=(8, 4))
        plt.xlabel("Prediction")
        plt.ylabel("Target")
        plt.figtext(0, 0, f"RMSE: {self.test_rmse}", fontsize=13)
        plt.grid()
        plt.scatter(x=Trainer.y_true, y=Trainer.y_pred, c='b', s=1)

        win = tk.Toplevel()
        win.wm_title("Window")
        win.geometry("1000x500")

        # specify the window as master
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky=tk.W)

        # navigation toolbar
        toolbarFrame = tk.Frame(master=win)
        toolbarFrame.grid(row=1, column=0)
        toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)

    def train_button_action(self):
        """ Takes in the hyperparameters from the model and trains the RSHDMRGPR """
        if DataLoader.data is None:
            messagebox.showerror("Information", "Data file is empty, please load the data first.")
            return

        m, n = DataLoader.data.shape

        scale = DataLoader.data['out'].max() - DataLoader.data['out'].min()
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(DataLoader.data)
        data_scaled = pd.DataFrame(scaler.transform(DataLoader.data), columns=DataLoader.data.columns)

        # Model hyperparameters
        d = int(self.hdmr_dim_text.get())
        if d <= 0 or d >= n:
            messagebox.showerror("Information", f"Please enter a valid dimension for HDMR to train. The dimension "
                                                f"has to be a positive integer less than the number of features"
                                                f"the data has which is {n - 1}.")
            return

        # creates the matrices and kernels used for RSHDMRGPR
        matrices, kernels = kernel_matrices(d, n - 1, kernel_function=RBF,
                                            length_scale=float(self.length_scale_text.get()))

        Trainer.model = RSHDMRGPR(matrices, kernels)

        # Sets the training hyperparameters
        alpha = float(self.alpha_text.get())
        n_restarts = int(self.n_restarts_text.get())
        cycles = int(self.cycle_text.get())
        scale_down = (float(self.scale_down1_text.get()), float(self.scale_down2_text.get()))
        optimizer = None if self.var.get() == 0 else 'fmin_l_bfgs_b'
        opt_every = int(self.opt_every_text.get())

        # Splits the data into training and testing
        x_train, x_test, y_train, y_test = train_test_split(data_scaled.drop(columns=['out']),
                                                            data_scaled['out'],
                                                            train_size=int(self.train_size_text.get()))
        print(x_train.shape, x_test.shape)

        # Trains the model
        Trainer.model.train(x_train, y_train, alphas=alpha, n_restarts=n_restarts, cycles=cycles,
                            scale_down=scale_down, optimizer=optimizer, opt_every=opt_every, verbose=2)

        Trainer.y_pred = batch_predict(Trainer.model, data_scaled.drop(columns=['out']))
        Trainer.y_true = data_scaled['out']

        self.test_rmse = scale * math.sqrt(mean_squared_error(Trainer.y_pred, Trainer.y_true))
        print(self.test_rmse)
        self.r_squared = np.corrcoef(Trainer.y_pred * scale, data_scaled['out'] * scale)[0, 1] ** 2
        print(self.r_squared)

        models = Trainer.model.get_models()
        param_string = f'Component Function Trained Parameters:\n'
        for i in range(len(models)):
            param_string += "length scale: {:.4f}".format(models[i].kernel_.k1.length_scale) + ' ' + \
                            "noise level: {:.4e}".format(models[i].kernel_.k2.noise_level) + '\n'
        param_string += f'\nRMSE on the test set: {self.test_rmse}\n'
        param_string += f'R^2 value on the test set: {self.r_squared}'
        display_params = ttk.Label(self, text=param_string, width=40)
        display_params.grid(row=24 + 7, column=0, columnspan=2, sticky=tk.W)

    def load_and_predict(self):
        """ Load a model and use it to predict on the data """
        if DataLoader.data is None:
            messagebox.showerror("Information", "Data file is empty, please load the data first.")
            return

        path = filedialog.askopenfilename()
        with open(path, 'rb') as file:
            Trainer.model = pickle.load(file)

        scale = DataLoader.data['out'].max() - DataLoader.data['out'].min()
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(DataLoader.data)
        data_scaled = pd.DataFrame(scaler.transform(DataLoader.data), columns=DataLoader.data.columns)

        Trainer.y_pred = batch_predict(Trainer.model, data_scaled.drop(columns=['out']))
        Trainer.y_true = data_scaled['out']

        self.test_rmse = scale * math.sqrt(mean_squared_error(Trainer.y_pred, Trainer.y_true))
        print(self.test_rmse)
        self.r_squared = np.corrcoef(Trainer.y_pred * scale, data_scaled['out'] * scale)[0, 1] ** 2
        print(self.r_squared)

        models = Trainer.model.get_models()
        param_string = f'Component Function Trained Parameters:\n'
        for i in range(len(models)):
            param_string += "length scale: {:.4f}".format(models[i].kernel_.k1.length_scale) + ' ' + \
                            "noise level: {:.4e}".format(models[i].kernel_.k2.noise_level) + '\n'
        param_string += f'\nRMSE on the test set: {self.test_rmse}\n'
        param_string += f'R^2 value on the test set: {self.r_squared}'
        display_params = ttk.Label(self, text=param_string, width=40)
        display_params.grid(row=24 + 7, column=0, columnspan=2, sticky=tk.W + tk.E)

    def save_model(self):
        """ Save a model """
        if Trainer.model is None:
            messagebox.showerror("Information", "No model has been trained. Please train a model first.")
            return

        path = filedialog.asksaveasfile(mode='wb', defaultextension='.pkl')
        with open(path.name, "wb") as file:
            pickle.dump(Trainer.model, file)

    def save_prediction(self):
        """ Saves the prediction along with the dataset """
        if DataLoader.data is None:
            messagebox.showerror("Information", "Data file is empty, please load the data first.")
            return
        if Trainer.y_pred is None:
            messagebox.showerror("Information", "Preciction has not been made, please train a new model and predict or "
                                                "load a model and predict.")
            return

        path = filedialog.asksaveasfile(mode='w', defaultextension=".csv",  filetypes=[("csv files", '*.csv'),
                                                                                       ("xlsx files", '*.xlsx'),
                                                                                       ("dat files", '*.dat')])

        copy_data = DataLoader.data.copy()
        copy_data['prediction'] = Trainer.y_pred
        copy_data.to_csv(path, index=False)

        # Clears memory
        copy_data.drop(copy_data.index, inplace=True)
        del copy_data


class UIApplication(tk.Tk):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.grid_propagate(0)

        self.title("RSHDMRGPR GUI")   # Title of the main window
        self.geometry("1400x800")  # Size of the main window
        self.resizable(width=True, height=True)  # Allows for resizing the main window

        #####################################################################################
        # Adds the data loader component
        #####################################################################################

        DataLoader(self).grid(row=0, column=0, sticky="nsew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=4)

        #####################################################################################
        # Adds the training component
        #####################################################################################

        Trainer(self).grid(row=0, column=1, sticky="nsew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)


if __name__ == '__main__':
    app = UIApplication()
    app.mainloop()
