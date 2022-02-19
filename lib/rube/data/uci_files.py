import logging
import os
import pathlib

import pandas as pd


EXCEL_COLUMNS = {'Invoice', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'Price', 'Customer ID', 'Country'}
REMOTE_FILE = "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx"
SHEET_NAME = "Year 2009-2010"

# default place to store local copies is alongside RUBE/
UCI_DATA_DIR = pathlib.Path(__file__).absolute().parents[4] / 'RUBE_data'


def load_uci_file(infile, sheet):
    """
    :param infile: the idea is to set this to REMOTE_FILE
    :param sheet: again, the idea is to apply SHEET_NAME here
    :return: pandas dataframe containing the data.
    Note there is flexibility to use a separately-saved csv version of the data
    """
    logging.info('Loading ' + infile + ' , sheet ' + sheet)
    if infile.endswith('xlsx'):
        datf = pd.read_excel(infile, sheet_name=sheet)
    elif infile.endswith('csv'):
        sheet = "number one, obviously"
        datf = pd.read_csv(infile)
        datf['InvoiceDate'] = pd.to_datetime(datf.InvoiceDate)
        del datf['Unnamed: 0']
    else:
        raise NotImplementedError
    assert EXCEL_COLUMNS <= set(datf.to_dict().keys())
    logging.info('Loaded ' + infile + ' , sheet ' + sheet)
    return datf


def standard_uci_data_access(data_dir):
    local_data_file = os.path.join(data_dir, 'uci_raw.csv')
    if os.path.exists(local_data_file):
        return load_uci_file(local_data_file, SHEET_NAME)
    df = load_uci_file(REMOTE_FILE, SHEET_NAME)
    df.to_csv(local_data_file)
    logging.info('Saving a copy to ' + local_data_file)
    return df
