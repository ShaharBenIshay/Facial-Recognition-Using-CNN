import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import trainer
import logger

# Note: need to use same configuration in trainer.py
logger.configure_logging_experiments()

# Configure evaluation results path
timestamp = pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')
evaluation_path = f'results/evaluation_{timestamp}.csv'

# Fixed hyper params:
validation_size = 0.2
optimizer = 'Adam'

# Hyper params to tune:
batch_sizes = [32]
learning_rates = [0.005]
regularization_lambdas = [0]
num_epochs = [25]
dropout_rates = [0]

hyperparameter_combinations = list(itertools.product(batch_sizes, learning_rates, regularization_lambdas, num_epochs,
                                                     dropout_rates))
evaluation_dict = {
    'Experiment ID': [],
    'Num Epochs': [],
    'Batch Size': [],
    'Learning Rate': [],
    'Regularization Lambda': [],
    'DropOut': [],
    'Train Time': [],
    'Train Loss Value': [],
    'Validation Accuracy': [],
    'Validation Loss': [],
    'Test Time': [],
    'Test Loss': [],
    'Test Accuracy': []
}

print(f'Starting hyperparams experiments - total number of experiments: {len(hyperparameter_combinations)}')
logging.info(f'Starting hyperparams experiments - total number of experiments: {len(hyperparameter_combinations)}')
experiment_id = 1
for hyperparameter_combination in hyperparameter_combinations:
    logging.info(f'Starting Experiment #{experiment_id}')
    batch_size, lr, reg_lambda, num_epoch, dropout = hyperparameter_combination
    evaluation_dict['Experiment ID'].append(experiment_id)
    evaluation_dict['Batch Size'].append(batch_size)
    evaluation_dict['Num Epochs'].append(num_epoch)
    evaluation_dict['Learning Rate'].append(lr)
    evaluation_dict['Regularization Lambda'].append(reg_lambda)
    evaluation_dict['DropOut'].append(dropout)

    current_trainer = trainer.Trainer(batch_size=batch_size, optimizer_name=optimizer, learning_rate=lr,
                                      num_epochs=num_epoch, validation_size=validation_size,
                                      regularization_lambda=lr, use_gpu_flag=True,
                                      dropout_rate=dropout, experiment_idx=experiment_id)

    train_time = current_trainer.train()
    test_time = current_trainer.test()
    test_accuracy = current_trainer.test_accuracy
    test_loss = current_trainer.test_loss
    validation_accuracy = current_trainer.validation_accuracy
    validation_loss = current_trainer.validation_loss
    current_train_loss_value = current_trainer.train_loss_value
    evaluation_dict['Train Time'].append(train_time)
    evaluation_dict['Train Loss Value'].append(round(current_train_loss_value, 3))
    evaluation_dict['Validation Accuracy'].append(round(validation_accuracy, 3))
    evaluation_dict['Validation Loss'].append(round(validation_loss, 3))
    evaluation_dict['Test Time'].append(test_time)
    evaluation_dict['Test Loss'].append(round(test_loss, 3))
    evaluation_dict['Test Accuracy'].append(round(test_accuracy, 3))
    logging.info(f'Finished Experiment #{experiment_id}')
    experiment_id += 1

logging.info('Finished all experiments')
evaluation_df = pd.DataFrame.from_dict(evaluation_dict)
evaluation_df.to_csv(evaluation_path)
logging.info('Saved evaluation statistics to CSV')


def report_plots(x_varying_values, test_acc_values, validation_acc_values, varying_values_str):
    """
    Create plots for extra experiments on best model - #4
    """
    TEST_COLOR = 'blue'
    VALIDATION_COLOR = 'red'
    best_validation_acc_idx = np.argmax(validation_acc_values)
    best_test_acc_idx = np.argmax(test_acc_values)
    plt.annotate('Max Test Accuracy', xy=(x_varying_values[best_test_acc_idx],
                                          test_acc_values[best_test_acc_idx]),
                 xytext=(x_varying_values[best_test_acc_idx] + 1,
                         test_acc_values[best_test_acc_idx] + 0.5), arrowprops=dict(facecolor='black', shrink=0.05))

    plt.annotate('Max Validation Accuracy', xy=(x_varying_values[best_validation_acc_idx],
                                                validation_acc_values[best_validation_acc_idx]),
                 xytext=(x_varying_values[best_validation_acc_idx] + 1,
                         validation_acc_values[best_validation_acc_idx] + 0.5),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    plt.plot(x_varying_values, test_acc_values, color=TEST_COLOR, label='Test')
    plt.plot(x_varying_values, validation_acc_values, color=VALIDATION_COLOR, label='Validation')
    plt.scatter(x_varying_values, test_acc_values, color=TEST_COLOR, marker='o')
    plt.scatter(x_varying_values, validation_acc_values, color=VALIDATION_COLOR, marker='o')
    plt.xlabel(f'{varying_values_str}')
    plt.ylabel('Accuracy')
    plt.title(f'Growing {varying_values_str} VS Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


########## Extra experiments for best model:
experiment = ''  # Configure experiment type

if experiment == 'Batch':
    # run for 25 epochs
    x_batches = evaluation_dict['Batch Size']  # [8, 16, 32, 64]
    y_test_acc_batches = evaluation_dict['Test Accuracy']
    y_validation_acc_batches = evaluation_dict['Validation Accuracy']
    report_plots(x_batches, y_test_acc_batches, y_validation_acc_batches, 'Number Of Batches')

if experiment == 'LR':
    # run for 25 epochs
    x_lr = evaluation_dict['Learning Rate']  # [0.0001, 0.0005, 0.001, 0.005]
    y_test_acc_lr = evaluation_dict['Test Accuracy']
    y_validation_acc_lr = evaluation_dict['Validation Accuracy']
    report_plots(x_lr, y_test_acc_lr, y_validation_acc_lr, 'Learning Rate')

if experiment == 'Epochs':
    x_epochs = evaluation_dict['Num Epochs']  # [5, 15, 25, 50]
    y_test_acc_epochs = evaluation_dict['Test Accuracy']
    y_validation_acc_epochs = evaluation_dict['Validation Accuracy']
    report_plots(x_epochs, y_test_acc_epochs, y_validation_acc_epochs, 'Number Of Epochs')

