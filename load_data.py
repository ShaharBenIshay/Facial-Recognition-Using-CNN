import numpy as np
import os
import pickle
import tqdm
from PIL import Image
import logging
import logger
import csv
from collections import Counter

logger.configure_logging_trainer()


def open_and_resize_img(img_path, width=105, height=105):
    """
    Open an image from a given path and resize it to the given width and height
    :return: the image as a numpy array
    """
    # Check if image in image path is jpg file, raise exception if not
    if not img_path.endswith('.jpg') and not img_path.endswith('.jpeg'):
        raise Exception(f'Image in {img_path} is not a jpg or jpeg file')
    img = Image.open(img_path)  # img_path end is jpg file
    img = img.resize((width, height))  # to 105 X 105 as paper
    data = np.asarray(img)
    data = np.array(data, dtype='float64')
    return data


def modify_img_name(person_name, img_num):
    """
    Modify the image name to the format: person_name_img_num.jpg
    :return: the modified image file name
    """
    base = '0000'
    offset = len(base)
    img_number = base + img_num  # omri, 1 -> omri_0001.jpg
    img_file_name = f'{person_name}_{img_number[-offset:]}.jpg'
    return img_file_name


def img_to_array(data_img_path, person_name_dir, image_num, width=105, height=105, channels=1, for_train=True):
    """
    Open and resize an image from the given img_path and return it as a numpy array
    :param data_img_path: path to the images' directory
    :param person_name_dir: the directory according to the name of the person
    :param image_num: the number of the image of the person
    :param for_train: default True - for reshaping specifically for train set
    :return:
    """
    img_file_name = modify_img_name(person_name_dir, image_num)
    image_path = os.path.join(data_img_path, person_name_dir, img_file_name)
    image_data = open_and_resize_img(image_path, width, height)
    # Change shape for train to add channels
    if for_train:
        return image_data.reshape(width, height, channels)
    else:  # for test
        return image_data


def handle_pairs(data_img_path, line, x_n1, x_n2, y, width=105, height=105, cells=1, is_matched=True):
    """
    Handle the pairs_txt of images in the train and test sets
    :param data_img_path: path to the images' directory
    :param line: the line in the train or test set
    :param x_n1: list of the first images in the pairs_txt
    :param x_n2: list of the second images in the pairs_txt
    :param y: list of the labels of the pairs_txt
    :param is_matched: boolean value to indicate if the pairs_txt are matched or mismatched
    :return: the lists of the first images, the second images and the labels of the pairs_txt
    """
    if is_matched:
        person_name, n1_image_num, n2_image_num = line
        n1_img = img_to_array(data_img_path, person_name, n1_image_num, width, height, cells)
        n2_img = img_to_array(data_img_path, person_name, n2_image_num, width, height, cells)
    else:  # mismatched
        n1_person_name, n1_image_num, n2_person_name, n2_image_num = line
        n1_img = img_to_array(data_img_path, n1_person_name, n1_image_num, width, height, cells)
        n2_img = img_to_array(data_img_path, n2_person_name, n2_image_num, width, height, cells)

    x_n1.append(n1_img)
    x_n2.append(n2_img)
    y.append(1 if is_matched else 0)
    return x_n1, x_n2, y


def load_and_process_data(dataset_path, dataset_name, data_img_path, path_to_save):
    """
    Load and process a dataset, to be ready for the model
    :param dataset_path: path to the datasets' directory
    :param dataset_name: the name of the dataset, train or test
    :param data_img_path: path to the images' directory
    :param path_to_save: path to save the processed data
    """
    txt_file_path = os.path.join(dataset_path, f'{dataset_name}.txt')
    width, height, cells = 105, 105, 1  # setup values
    x_n1, x_n2, y = [], [], []
    ground_truth = []  # list of the original line in the data
    with open(txt_file_path, 'r') as file:
        file_lines = file.readlines()  # read the file lines
    for line_as_str in tqdm.tqdm(file_lines):
        line = line_as_str.split()  # line is now a list []
        is_matched_pair = len(line) == 3
        if not 3 <= len(line) <= 4:  # not 3 or 4
            logging.warning(f'Encountered an invalid line: {line}')
            continue
        # line is valid:
        ground_truth.append(line)
        if is_matched_pair:  # match = same person
            x_n1, x_n2, y = handle_pairs(data_img_path, line, x_n1, x_n2, y, width, height, cells, is_matched=True)
        else:  # not is_matched_pair -> mismatch  two different persons
            x_n1, x_n2, y = handle_pairs(data_img_path, line, x_n1, x_n2, y, width, height, cells, is_matched=False)
    # write the processed data
    with open(path_to_save, 'wb') as f:
        pickle.dump([x_n1, x_n2, y, ground_truth], f)


def dataset_analysis():
    """
    Counts number of match/mismatched pairs in train/test sets
    """
    with open(f'data/pairs_txt/pairsDevTrain.txt', 'r') as train_txt:
        train = list(csv.reader(train_txt, delimiter='\t'))
    with open(f'data/pairs_txt/pairsDevTest.txt', 'r') as test_txt:
        test = list(csv.reader(test_txt, delimiter='\t'))
    # train_label_change_line, test_label_change_line = train[0], test[0]
    train, test = train[1:], test[1:]

    analysis_dict = {'Train set size': len(train),
                     'Test set size': len(test)}

    train_match_mismatch_counts = Counter(['match' if len(pair) == 3 else 'mismatch' for pair in train])
    analysis_dict['Number of matched pairs in train-set'] = train_match_mismatch_counts['match']
    analysis_dict['Number of mismatched pairs in train-set'] = train_match_mismatch_counts['mismatch']

    test_match_mismatch_counts = Counter(['match' if len(pair) == 3 else 'mismatch' for pair in test])
    analysis_dict['Number of matched pairs in test-set'] = test_match_mismatch_counts['match']
    analysis_dict['Number of mismatched pairs in test-set'] = test_match_mismatch_counts['mismatch']

    for k, v in analysis_dict.items():
        print(f'{k}: {v}')

    # Count number of total pictures in data/lfw2/lfw2 directory
    num_pictures = 0
    for person_name in os.listdir('data/lfw2/lfw2'):
        num_pictures += len(os.listdir(f'data/lfw2/lfw2/{person_name}'))
    print(f'\nTotal number of pictures in data/lfw2/lfw2: {num_pictures}')

