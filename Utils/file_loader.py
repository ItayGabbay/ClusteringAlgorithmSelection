import os
from glob import glob
import pandas as pd

def get_dataframes():
    PATH = os.path.join(os.getcwd(), 'Datasets')
    EXT = '*csv'
    all_csv_files = [file
                     for path, subdir, files in os.walk(PATH)
                     for file in glob(os.path.join(path, EXT))]
                     
    return all_csv_files

def get_generated_dataframes():
    PATH = os.path.join(os.getcwd(), 'Generated')
    EXT = '*csv'
    all_csv_files = [file
                     for path, subdir, files in os.walk(PATH)
                     for file in glob(os.path.join(path, EXT))]
                     
    return all_csv_files