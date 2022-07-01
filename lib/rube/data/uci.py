import os

import pandas as pd
import logging
import datetime

from rube.data import uci_files
from rube.data.real import RealDataGenerator


class UCIGenerator(RealDataGenerator):
    def import_dataset(self, begin_week=0, end_week=None, n_lines=None):
        return import_data(begin_week=begin_week, end_week=end_week, n_lines=n_lines)


def import_data(begin_week=0, end_week=None, n_lines=None):
    df = uci_files.standard_uci_data_access(uci_files.UCI_DATA_DIR)
    # clean data:
    invalids = invalid_series(df)
    data = df[~invalids]
    data = augment_time(data)
    if n_lines:
        data = data.head(n_lines)
    if not end_week:
        end_week = data['Week'].max()
    logging.info(f"We now remove all data except in weeks {begin_week} to {end_week} of the dataset (inclusive).")
    end_week += 1
    data = data[data['Week'].isin(range(begin_week, end_week))]
    return data


def is_invalid(datapoint):
    """
    :param datapoint: a datapoint from the UCI opensource data
    we weed out negative quantities and zero prices.
    There are a bunch of duff Stock Codes corresponding to postage, etc.
    There are also a small number of idiosyncratic cases: 'DCGS0066N', 'DCGSSBOY', 'DCGSSGIRL', 'SP1002'
    Sometimes, the description contains information indicating that this is not describing a real purchase
    :return: whether this is a valid datapoint to process into the raw files
    """

    if datapoint['Quantity'] < 0:
        return True

    if datapoint['Price'] == 0:
        return True

    for s in ('POST', 'DOT', 'gift', 'TEST', 'BANK CHARGES', 'ADJUST',
              'PADS', 'AMAZONFEE', 'DCGS0066N', 'DCGSSBOY', 'DCGSSGIRL', 'SP1002'):
        if s in str(datapoint['StockCode']):
            return True

    if datapoint['Description'] in ('Discount', 'Manual', 'SAMPLES', 'Adjust bad debt', 'update'):
        return True

    return False


def invalid_series(datf, local_data_file=None):
    """
    :param datf:
    :return: boolean Series saying whether each item in the dataframe is_invalid()
    """
    local_data_file = local_data_file or uci_files.UCI_DATA_DIR / 'invalids.csv'
    if os.path.exists(local_data_file):
        in_data = pd.read_csv(local_data_file)
        return in_data[in_data.columns[-1]]
    df = pd.Series([is_invalid(row) for index, row in datf.iterrows()])
    df.to_csv(local_data_file)
    logging.info('Saving a copy to ' + str(local_data_file))
    return df


def augment_time(df):
    df['Hour'] = df.InvoiceDate.dt.hour
    df['Month'] = df.InvoiceDate.dt.month
    df['Year'] = df.InvoiceDate.dt.year

    df['Week'] = (df.InvoiceDate - datetime.datetime(1970, 1, 1)).dt.days // 7
    df['Week'] = df.Week - df.Week.min()
    return df

