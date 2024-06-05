import copy
import logging
import pickle
import time
from tqdm import tqdm
import logger
import torch.nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim
from torch.optim.lr_scheduler import StepLR
from siamese_network import *
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Configure logging
# logger.configure_logging_trainer()
logger.configure_logging_experiments()


class Trainer(object):
    """
    Trainer class for training and testing the Siamese Network
    """

    def __init__(self, num_epochs, batch_size, optimizer_name, learning_rate, regularization_lambda=0.0,
                 dropout_rate=0.0, validation_size=0.2, experiment_idx=0, use_gpu_flag=False, use_early_stopping=False):

        self.seed = 42  
        self.model_use_gpu = use_gpu_flag
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.epochs = num_epochs
        self.batch_size = batch_size
        self.optimizer_name = optimizer_name  # is a string
        self.learning_rate = learning_rate
        self.regularization_lambda = regularization_lambda
        self.dropout = dropout_rate
        self.prediction_threshold = 0.5
        self.siamese_model = None
        self.train_loss_func = None
        self.scheduler = None
        self.use_early_stopping = use_early_stopping
        self.patience = 5
        self.validation_size = validation_size
        self.train_loss_value = 0.0
        self.validation_loss = 0.0
        self.validation_accuracy = 0.0
        self.test_loss = 0.0
        self.test_accuracy = 0.0
        self.experiment_idx = experiment_idx

        logging.info(
            f'Initialized Trainer with params: epochs={self.epochs}, batch size={self.batch_size}, '
            f' validation={self.validation_size},'
            f'\n    optimizer={self.optimizer_name}, learning rate={self.learning_rate} , '
            f'      regularization lambda={self.regularization_lambda}, '
            f'\n        dropout={self.dropout}, early stopping={self.use_early_stopping}')

    def prepare_input(self, x):
        """
        Prepare the input data before creating a dataset to pass through the Siamese Network
        :param x: the input data we load from the pkl files
        :return: x: the prepared input data
        """
        x = np.array(x)  # for faster conversion to tensor
        x = torch.tensor(x).float()
        x = x.unsqueeze(1).squeeze(-1)
        return x

    def prepare_train_datasets(self):
        """
        Create training and validation datasets for the Siamese Network as PyTorch Dataset objects
        :return: PyTorch Datasets: train_dataset, validation_dataset
        """
        with open('data/processed_train.pkl', 'rb') as f:
            x1_train, x2_train, y_train, _ = pickle.load(f)

        x1_train = self.prepare_input(x1_train)
        x2_train = self.prepare_input(x2_train)
        y_train = torch.tensor(y_train)

        # Train-Validation Split
        x_n1_train, x_n1_validation, y_n1_train, y_n1_validation = \
            train_test_split(x1_train, y_train, test_size=self.validation_size, random_state=self.seed)
        x_n2_train, x_n2_validation, y_n2_train, y_n2_validation = \
            train_test_split(x2_train, y_train, test_size=self.validation_size, random_state=self.seed)
        # Check if the train-validation split succeeded
        train_sets_equal = torch.eq(y_n1_train, y_n2_train)
        validation_sets_equal = torch.eq(y_n1_validation, y_n2_validation)
        equal_sets_condition = train_sets_equal.all() or validation_sets_equal.all()
        if not equal_sets_condition:
            raise Exception('Train-Validation Split did not succeed')

        # Set Y data explicitly for train and validation datasets and create PyTorch Datasets
        y_train_dataset = y_n1_train
        train_dataset = TensorDataset(x_n1_train, x_n2_train, y_train_dataset)
        y_validation_dataset = y_n1_validation
        validation_dataset = TensorDataset(x_n1_validation, x_n2_validation, y_validation_dataset)
        return train_dataset, validation_dataset

    def prepare_test_datasets(self):
        """
        Create test dataset for the Siamese Network as a PyTorch Dataset object
        :return: PyTorch Dataset: the test dataset
        """
        with open('data/processed_test.pkl', 'rb') as f:
            x1_test, x2_test, y_test, _ = pickle.load(f)

        x1_test = self.prepare_input(x1_test)
        x2_test = self.prepare_input(x2_test)
        y_test = torch.tensor(y_test)
        test_dataset = TensorDataset(x1_test, x2_test, y_test)
        return test_dataset

    def _prob_to_binary_decision(self, y_probs):
        """
        Convert the predicted probabilities to binary decisions - 0 (mismatch) or 1 (match)
        """
        y_binaries = []
        for y_prob in y_probs:
            if y_prob > self.prediction_threshold:
                y_binaries.append(1)
            else:
                y_binaries.append(0)
        return torch.tensor(y_binaries)

    def train(self):
        """
        Train the Siamese Network
        :return: train_time: the time it took to train the model
        """
        # Create DataLoaders from PyTorch Datasets
        train_dataset, validation_dataset = self.prepare_train_datasets()
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize the Siamese Network
        self.siamese_model = SiameseNetwork(use_gpu_flag=self.model_use_gpu, dropout_rate=self.dropout)
        # Choose optimizer
        if self.optimizer_name == "SGD":  # option to choose SGD
            optimizer = torch.optim.SGD(self.siamese_model.parameters(), lr=self.learning_rate,
                                        weight_decay=self.regularization_lambda)
        else:  # default optimizer = Adam
            optimizer = torch.optim.Adam(self.siamese_model.parameters(), lr=self.learning_rate,
                                         weight_decay=self.regularization_lambda)
        self.scheduler = StepLR(optimizer, step_size=5, gamma=0.95)  # default gamma is 0.1 by documentation

        if self.model_use_gpu:  # Move model to GPU
            self.siamese_model.to(self.device)

        loss_function = torch.nn.BCELoss()  # Set loss function

        # Early stopping settings
        patience_counter = self.patience
        best_validation_accuracy, best_train_epoch = 0, 0
        last_train_epoch = self.epochs

        # Training Loop
        print(f'Start Training Loop for {self.epochs} epochs, with batches of size {self.batch_size}')
        logging.info(f'Start Training Loop for {self.epochs} epochs, with batches of size {self.batch_size}')

        # Save epochs' run results to plot later
        train_loss_vs_epochs, validation_loss_vs_epochs, validation_acc_vs_epochs = [], [], []

        train_epochs_progress_bar = tqdm(iterable=range(1, self.epochs + 1), total=self.epochs, desc='Training Epochs',
                                         position=0, leave=True)
        training_start_time = time.time()
        for epoch_idx in train_epochs_progress_bar:
            self.train_loss_value = 0.0  # Reset train loss (for current epoch)
            # Training Step
            print(f'Train Epoch #{epoch_idx}')
            logging.info(f'Train Epoch #{epoch_idx}')
            self.siamese_model.train()
            train_loader_progress_bar = tqdm(iterable=enumerate(train_loader), total=len(train_loader),
                                             desc='Training Batches', position=1, leave=True, colour='green')
            for batch_idx, (curr_batch_x1, curr_batch_x2, curr_batch_y) in train_loader_progress_bar:
                if self.model_use_gpu:
                    curr_batch_x1 = curr_batch_x1.to(self.device)
                    curr_batch_x2 = curr_batch_x2.to(self.device)
                    curr_batch_y = curr_batch_y.to(self.device)
                y_pred = self.siamese_model(curr_batch_x1, curr_batch_x2)
                curr_batch_y = curr_batch_y.unsqueeze(1).float()
                self.train_loss_func = loss_function(y_pred, curr_batch_y)
                # Update Parameters Step
                optimizer.zero_grad()
                self.train_loss_func.backward()
                optimizer.step()
                self.train_loss_value += self.train_loss_func.item()

            self.scheduler.step()  # Every 5 epochs perform a step: lr = lr * gamma

            self.train_loss_value /= len(train_loader)  # Calculate average train loss for current epoch
            train_loss_vs_epochs.append((epoch_idx, self.train_loss_value))

            # Validation Step
            self.siamese_model.eval()
            # Reset validation loss and accuracy (for current epoch)
            self.validation_loss, self.validation_accuracy = 0.0, 0.0
            with (torch.no_grad()):  # No gradient calculation in validation step
                validation_loader_progress_bar = tqdm(iterable=enumerate(validation_loader),
                                                      total=len(validation_loader), desc='Validation Batches',
                                                      position=2, leave=True, colour='blue')
                for batch_idx, (curr_batch_x1, curr_batch_x2, curr_batch_y) in validation_loader_progress_bar:
                    if self.model_use_gpu:
                        curr_batch_x1 = curr_batch_x1.to(self.device)
                        curr_batch_x2 = curr_batch_x2.to(self.device)
                        curr_batch_y = curr_batch_y.to(self.device)
                    y_pred = self.siamese_model(curr_batch_x1, curr_batch_x2)
                    curr_validation_loss = loss_function(y_pred.squeeze(-1), curr_batch_y.float())
                    self.validation_loss += curr_validation_loss.item()  # Collect validation loss

                    y_validation_pred = self._prob_to_binary_decision(y_pred).to(self.device)
                    curr_validation_acc = accuracy_score(curr_batch_y.to('cpu'), y_validation_pred.to('cpu'))
                    curr_validation_acc = torch.from_numpy(np.array(curr_validation_acc)).to(self.device)
                    self.validation_accuracy += curr_validation_acc.item()  # Collect validation accuracy

                self.validation_loss /= len(validation_loader)  # Calculate average validation loss for current epoch
                validation_loss_vs_epochs.append((epoch_idx, self.validation_loss))
                self.validation_accuracy /= len(validation_loader)  # Calculate average validation accuracy for current epoch
                validation_acc_vs_epochs.append((epoch_idx, self.validation_accuracy))
                print(self.validation_accuracy)
            if self.use_early_stopping:
                if self.validation_accuracy > best_validation_accuracy:
                    best_validation_accuracy = self.validation_accuracy
                    patience_counter = self.patience
                else:  # No improvement
                    patience_counter -= 1
                    if patience_counter == 0:
                        last_train_epoch = epoch_idx
                        print(f'Early stopping after epoch {last_train_epoch}')
                        logging.info(f'Early stopping after epoch {last_train_epoch}')
                        break

        training_finish_time = time.time()
        train_time = training_finish_time - training_start_time
        print(
            f'Finished Training after {last_train_epoch} epochs -- last validation accuracy= {self.validation_accuracy}')
        logging.info(
            f'Finished Training after {last_train_epoch} epochs -- last validation accuracy= {self.validation_accuracy}')
        plot_value_vs_epoch(train_loss_vs_epochs, value_name='Train Loss', experiment_idx=self.experiment_idx)
        plot_value_vs_epoch(validation_loss_vs_epochs, value_name='Validation Loss', experiment_idx=self.experiment_idx)
        plot_value_vs_epoch(validation_acc_vs_epochs, value_name='Validation Accuracy',
                            experiment_idx=self.experiment_idx)

        return train_time

    def test(self):
        """
        Test the Siamese Network
        :return: test_time: the time it took to test the model
        """
        # Create DataLoader from PyTorch Dataset
        test_dataset = self.prepare_test_datasets()
        test_loader = DataLoader(test_dataset, shuffle=False)  # Notice shuffle=False
        if self.model_use_gpu:
            self.siamese_model.cuda()
        y_test_pred_prob = torch.tensor([]).to(self.device)
        print(f'Start Testing over {len(test_loader)} records')
        logging.info(f'Start Testing over {len(test_loader)} records')
        test_start_time = time.time()
        self.siamese_model.eval()
        with torch.no_grad():  # No gradient calculation in Test step
            test_loader_progress_bar = tqdm(iterable=enumerate(test_loader), total=len(test_loader),
                                            desc='Test Iterations')
            for i, (curr_x1, curr_x2, curr_y) in test_loader_progress_bar:
                if self.model_use_gpu:
                    curr_x1 = curr_x1.to(self.device)
                    curr_x2 = curr_x2.to(self.device)
                    curr_y = curr_y.to(self.device)
                y_pred = self.siamese_model(curr_x1, curr_x2)
                y_test_pred_prob = torch.cat((y_test_pred_prob, y_pred), dim=0)

            y_test_true = test_dataset.tensors[-1]
            y_test_pred_binary = self._prob_to_binary_decision(y_test_pred_prob)
            create_confusion_matrix(copy.copy(y_test_pred_prob), copy.copy(y_test_pred_binary), copy.copy(y_test_true))
            self.test_accuracy = accuracy_score(y_test_true, y_test_pred_binary)
            test_loss_func = torch.nn.BCELoss()
            self.test_loss = test_loss_func(y_test_pred_binary.float(), y_test_true.float()).item()
        test_finish_time = time.time()
        test_time = test_finish_time - test_start_time
        print(f'Finished Testing -- test accuracy= {self.test_accuracy}')
        logging.info(f'Finished Testing -- test accuracy= {self.test_accuracy}')
        return test_time


def create_confusion_matrix(y_pred_prob, y_pred_binary, y_true):
    """
    Create a confusion matrix and print the best and worse probabilities and records for TP, FP, TN, FN
    """
    actual_labels = y_true
    predicted_labels = y_pred_binary
    cm = confusion_matrix(actual_labels, predicted_labels, labels=[0, 1])
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Mismatch', 'Match'],
                yticklabels=['Mismatch', 'Match'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    ###########
    tp_max_prob, tp_argmax = 0, None
    fp_max_prob, fp_argmax = 0, None
    tn_min_prob, tn_argmin = 100, None
    fn_min_prob, fn_argmin = 100, None
    for idx in range(len(y_pred_prob)):
        true_label = y_true[idx]
        prob = y_pred_prob[idx]
        if prob > 0.5 and true_label == 1:  # TP
            if prob > tp_max_prob:
                tp_max_prob, tp_argmax = prob, idx
        if prob > 0.5 and true_label == 0:  # FP
            if prob > fp_max_prob:
                fp_max_prob, fp_argmax = prob, idx
        if prob <= 0.5 and true_label == 0:  # TN:
            if prob < tn_min_prob:
                tn_min_prob, tn_argmin = prob, idx
        if prob <= 0.5 and true_label == 1:  # FN
            if prob < fn_min_prob:
                fn_min_prob, fn_argmin = prob, idx

    print(f"TP best prob: {tp_max_prob.item()}, TP best index {tp_argmax}")
    print(f"FP worse prob: {fp_max_prob.item()}, FP worse index {fp_argmax}")
    print(f"TN best prob: {tn_min_prob.item()}, TN worse index {tn_argmin}")
    print(f"FN worse prob: {fn_min_prob.item()}, FP worse index {fn_argmin}")


def plot_value_vs_epoch(run_results, value_name, experiment_idx=0):
    """
    Plot the performance value vs. the run number
    :param run_results: list: the results of the runs, in the format [(epoch1, value1), (epoch2, value2), ...]
    :param: value_name: str: either loss or accuracy
    :param: experiment_idx: int: the index of the experiment
    """
    # Convert names name to lower case and replace whitespaces with underscores
    value_type = value_name.lower().replace(' ', '_')
    plt.figure()
    plt.plot(*zip(*run_results))
    plt.xlabel('Epochs')
    plt.ylabel(value_name)
    plt.title(f'{value_name} vs. Epochs')
    plt.savefig(f'results/plots/EXP_{experiment_idx}_{value_type}_vs_epochs.png')
    plt.close()

