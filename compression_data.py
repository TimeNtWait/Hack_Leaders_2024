import pandas as pd
import numpy as np
from pandas.api.types import is_float_dtype, is_integer_dtype
from datetime import datetime
from dateutil.relativedelta import relativedelta

import glob
import pyarrow.parquet as pq
from tqdm import trange, tqdm
from typing import List, Optional
import warnings

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')


class CompressDF:
    def __init__(self, ):
        pass

    # Загрузка списка файлов (типа паркет) в один датафрейм
    def load_df_by_files(self, files: list[str]) -> pd.DataFrame:
        union_df = pd.DataFrame()
        for file in tqdm(files):
            current_df = pq.read_table(file).to_pandas()
            union_df = pd.concat([union_df, current_df])
        return union_df

    # Определяем минимальный целочисленный тип
    def series_to_int(self, col_df: pd.Series) -> pd.Series:
        min_val = col_df.min()
        max_val = col_df.max()
        if min_val >= -128 and max_val <= 127:
            col_df = col_df.astype('int8')
        elif min_val >= -32768 and max_val <= 32767:
            col_df = col_df.astype('int16')
        elif min_val >= -2147483648 and max_val <= 2147483647:
            col_df = col_df.astype('int32')
        else:
            col_df = col_df.astype('int64')
        return col_df

    # Уменьшение размеров актуально для таргетов\, для транзакци и для фичей
    def compression(self, df: pd.DataFrame(), datetime_cols: List[str] = [], category_cols: List[str] = []):
        # лучше не переводить клиентов в категории, т.к. приходится дополнительно костылить
        # вызов fillna, что в итоге увеличивает время обработки
        float64_cols = list(df.select_dtypes(include='float64'))
        df[float64_cols] = df[float64_cols].astype('float32')
        for col in df.columns:
            if col in category_cols:
                df[col] = df[col].astype('category')
            elif col in datetime_cols:
                if df[col].dtypes == 'object':
                    df[col] = pd.to_datetime(df[col])
            # Если колонка содержит числа
            elif is_integer_dtype(df[col]):
                if df[col].dtypes == 'int8':
                    continue
                else:
                    df[col] = self.series_to_int(df[col])
            elif is_float_dtype(df[col]):
                # Возможно ли перевести в число
                if np.array_equal(df[col].fillna(0), df[col].fillna(0).astype(int)):
                    df[col] = df[col].fillna(0)
                    df[col] = self.series_to_int(df[col])
        return df

if __name__ == '__main__':
    print(f'Start ...')
    PATH = ''
    PATH_DATASET = PATH + 'datasets/sber_source/'
    PATH_DATASET_OUTPUT = PATH + 'datasets/'
    PATH_DATASET_TRX_TRAIN = PATH_DATASET + 'trx_train.parquet/'
    PATH_DATASET_TRX_TEST = PATH_DATASET + 'trx_test.parquet/'

    PATH_DATASET_TARGET_TRAIN = PATH_DATASET + 'train_target.parquet/'
    PATH_DATASET_TARGET_TEST = PATH_DATASET + 'test_target_b.parquet/'

    # Определяем пути к данным транзакциям
    train_trx_files = glob.glob(PATH_DATASET_TRX_TRAIN + '/*.parquet')
    test_trx_files = glob.glob(PATH_DATASET_TRX_TEST + '/*.parquet')

    trx_files = train_trx_files + test_trx_files

    # trx_files = train_target_files = glob.glob(PATH_DATASET_TARGET_TRAIN + '/*.parquet')
    # trx_files = train_target_files = glob.glob(PATH_DATASET_TARGET_TEST + '/*.parquet')
    # train_target_files = glob.glob(PATH_DATASET_TARGET_TRAIN + '/*.parquet')
    # test_target_files = glob.glob(PATH_DATASET_TARGET_TEST + '/*.parquet')
    #
    # trx_files = train_target_files + test_target_files

    compress = CompressDF()
    original_df = compress.load_df_by_files(files=trx_files)
    print(f'Загрузили данные')

    comress_df = compress.compression(df=original_df,
                                    datetime_cols=['mon'],
                                    # лучше не переводить клиентов в категории, т.к. приходится дополнительно костылить
                                    # вызов fillna, что в итоге увеличивает время обработки
                                    # category_cols=['client_id'],
                                )
    print(f'Сжали данные')
    # Сохраняем в файл оптимизированный файл
    # comress_df.to_parquet(PATH_DATASET_OUTPUT + 'compress_train_target_files_06_06_2024.parquet', compression='gzip')
    # comress_df.to_parquet(PATH_DATASET_OUTPUT + 'compress_test_target_files_08_06_2024.parquet')
    # comress_df.to_parquet(PATH_DATASET_OUTPUT + 'compress_targets_08_06_2024.parquet')
    comress_df.to_parquet(PATH_DATASET_OUTPUT + 'compress_train_target_files_11_06_2024.parquet', compression='gzip')
    print(f'Сохранили данные')
    print(f'Done!')
