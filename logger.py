import logging
from datetime import datetime

timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def configure_logging_experiments():
    logging.basicConfig(
        level=logging.INFO,  # Set the minimum level for log messages
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=f'logs/run_experiments_{timestamp}.log',  # Log file name
        filemode='w'  # 'w' for write, 'a' for append
    )

def configure_logging_trainer():
    logging.basicConfig(
        level=logging.INFO,  # Set the minimum level for log messages
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=f'logs/trainer.log',  # Log file name
        filemode='w'  # 'w' for write, 'a' for append
    )