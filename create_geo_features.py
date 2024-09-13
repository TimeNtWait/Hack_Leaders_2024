import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import glob
import pyarrow.parquet as pq
from typing import List


class GeoFeatures:
    def __int__(self, train_target_files: List[str], train_geo_files: List[str], test_geo_files: List[str]):
        # Скачиваем данные об успешных покупках продуктов
        self.train_target_df = load_df_by_files(train_target_files)
        # Загружаем все геоданные по трайн и тест
        self.all_geo_train_df = pd.concat([
            self.load_df_by_files(train_geo_files),
            self.load_df_by_files(test_geo_files)
        ])
        # Частота преобретения клиентом продуктов пользующихся определеным банкоматом
        self.union_geohash_4_df = pd.DataFrame()
        self.union_geohash_5_df = pd.DataFrame()
        self.union_geohash_6_df = pd.DataFrame()

        # Связка клиента и используемые им банкоматы в отчетный период
        self.union_geo_by_clients_4_df = pd.DataFrame()
        self.union_geo_by_clients_5_df = pd.DataFrame()
        self.union_geo_by_clients_6_df = pd.DataFrame()

        # Частота преобретения клиентом продуктов пользующихся определеным банкоматом
        self.clients_top_geohash_4_df = pd.DataFrame()
        self.clients_top_geohash_5_df = pd.DataFrame()
        self.clients_top_geohash_6_df = pd.DataFrame()


    # Загрузка списка файлов (типа паркет) в один датафрейм
    def load_df_by_files(files: list[str]) -> pd.DataFrame:
        union_df = pd.DataFrame()
        for file in files:
            current_df = pq.read_table(file).to_pandas()
            union_df = pd.concat([union_df, current_df])
        return union_df

    def calc_geohash_features(self, count_mon: int = 13):
        """
        Рассчитать частоту каждого геохэша - общую
        Частота преобретения клиентом продуктов пользующихся определеным банкоматом
        :param count_mon: - кол-во обрабатываемых месяцев
        :return: заполенные датафреймы по уровням 4/5/6
        """

        # Формируем статистику посещений каждого терминала уникальными клиентами и общее кол-во посещений
        start_date = datetime(2022, 1, 1, 0, 0, 0)
        # end_date = datetime(2023, 1, 1, 0, 0, 0)
        # Проходим по каждому отчетному периоду
        for i in range(count_mon):
            end_date = start_date + relativedelta(months=1) - relativedelta(days=1)
            print(f'start: {start_date}, end: {end_date}')

            select_mon_geo_df = self.all_geo_train_df[self.all_geo_train_df['event_time'].between(start_date, end_date)]
            print(select_mon_geo_df.shape)

            def calc_aggregate_by_geohash(geo_index: str = 'geohash_4'):
                geohash_df = select_mon_geo_df.groupby(geo_index).agg(
                    count_trx=('client_id', len),
                    uniq_clients=('client_id', pd.Series.nunique),
                )
                geohash_df['report_end'] = end_date
                geohash_df['report_next_end'] = start_date + relativedelta(months=2) - relativedelta(days=1)

                # Рассчитываем уровень "продоваемости продукта" относительно геопозиции
                train_client_by_cur_mon_df = train_target_df[pd.to_datetime(train_target_df['mon']).between(start_date,
                                                                                                            start_date + relativedelta(
                                                                                                                months=1) - relativedelta(
                                                                                                                days=1))]
                current_mon_train_df = train_client_by_cur_mon_df.rename(columns={'mon': 'mon_report'}).add_prefix(
                    'cur_mon_')

                client_geo_df = select_mon_geo_df[['client_id', geo_index]].drop_duplicates()
                cur_mon_select_geo_target_df = client_geo_df.merge(current_mon_train_df, left_on='client_id',
                                                                   right_on='cur_mon_client_id', how='left').fillna(0)

                popular_product_by_geohash_df = cur_mon_select_geo_target_df.groupby(geo_index).agg(
                    cur_sum_target_1=('cur_mon_target_1', sum),
                    cur_sum_target_2=('cur_mon_target_2', sum),
                    cur_sum_target_3=('cur_mon_target_3', sum),
                    cur_sum_target_4=('cur_mon_target_4', sum),

                    cur_median_target_1=('cur_mon_target_1', np.median),
                    cur_median_target_2=('cur_mon_target_2', np.median),
                    cur_median_target_3=('cur_mon_target_3', np.median),
                    cur_median_target_4=('cur_mon_target_4', np.median),

                    cur_var_target_1=('cur_mon_target_1', 'var'),
                    cur_var_target_2=('cur_mon_target_2', 'var'),
                    cur_var_target_3=('cur_mon_target_3', 'var'),
                    cur_var_target_4=('cur_mon_target_4', 'var'),

                    uniq_clients=('client_id', pd.Series.nunique),
                )

                # Расчитываем значение "популярности" геохешей относительно кол-ва клиентов
                columns = ['cur_sum_target_1', 'cur_sum_target_2', 'cur_sum_target_3', 'cur_sum_target_4',
                           'cur_median_target_1', 'cur_median_target_2', 'cur_median_target_3', 'cur_median_target_4',
                           'cur_var_target_1', 'cur_var_target_2', 'cur_var_target_3', 'cur_var_target_4', ]
                for col in columns:
                    popular_product_by_geohash_df[f'{col}__by_clients'] = popular_product_by_geohash_df[col] / \
                                                                          popular_product_by_geohash_df['uniq_clients']

                geohash_df = geohash_df.reset_index().merge(popular_product_by_geohash_df.reset_index(), on=geo_index,
                                                            how='left')
                return geohash_df

            geohash_4_df = calc_aggregate_by_geohash(geo_index='geohash_4').fillna(0)
            self.union_geohash_4_df = pd.concat([self.union_geohash_4_df, geohash_4_df])

            geohash_5_df = calc_aggregate_by_geohash(geo_index='geohash_5').fillna(0)
            self.union_geohash_5_df = pd.concat([self.union_geohash_5_df, geohash_5_df])

            geohash_6_df = calc_aggregate_by_geohash(geo_index='geohash_6').fillna(0)
            self.union_geohash_6_df = pd.concat([self.union_geohash_6_df, geohash_6_df])

            start_date = start_date + relativedelta(months=1)

    def calc_client_geo_features(self, count_mon: int = 13):
        """
        Связка клиента и используемые им банкоматы в отчетный период
        Частота преобретения клиентом продуктов пользующихся определеным банкоматом
        :param count_mon: - кол-во обрабатываемых месяцев
        :return: заполенные датафреймы по уровням 4/5/6
        """
        # Популярные геохеши у клиентов
        # Расчет топ-5 популярных хешей для каждого клиента, и также расчет процента посещения этих топ-5 относительно всех посещаемых геохешей (по аналогии с софтмакс)
        start_date = datetime(2022, 1, 1, 0, 0, 0)
        end_date = datetime(2023, 1, 1, 0, 0, 0)

        for i in range(count_mon):
            end_date = start_date + relativedelta(months=1) - relativedelta(days=1)
            print(f'start: {start_date}, end: {end_date}')

            select_mon_geo_df = self.all_geo_train_df[self.all_geo_train_df['event_time'].between(start_date, end_date)]
            print(select_mon_geo_df.shape)

            def calc_aggregate_client_by_geohash(geo_index: str = 'geohash_4'):
                client_geo_df = select_mon_geo_df.groupby(['client_id', geo_index]).size().reset_index(name='freq')

                geohash_df = client_geo_df.groupby(['client_id']).agg(
                    cnt_geo=('freq', len),
                    sum_geo_trx=('freq', sum),
                )

                geo_by_clients = geohash_df.reset_index().merge(client_geo_df, on='client_id', how='left')
                geo_by_clients['prc_use'] = geo_by_clients['freq'] / geo_by_clients['sum_geo_trx']

                geo_by_clients['report_end'] = end_date
                geo_by_clients['report_next_end'] = start_date + relativedelta(months=2) - relativedelta(days=1)

                popular_geo_by_clients = geo_by_clients.loc[
                    geo_by_clients.groupby('client_id')['prc_use'].transform('max').eq(geo_by_clients['prc_use'])]
                # Могут быть ситуации когда одинаковые геохеши имеют одинаковый вес, тогда выбираем первый
                popular_geo_by_clients = popular_geo_by_clients.drop_duplicates(subset=['client_id', 'prc_use'])

                popular_geo_by_clients['report_end'] = end_date
                popular_geo_by_clients['report_next_end'] = start_date + relativedelta(months=2) - relativedelta(days=1)

                return geo_by_clients.fillna(0), popular_geo_by_clients.fillna(0)

            geo_by_clients_4_df, pop_geo_by_clients_4_df = calc_aggregate_client_by_geohash(geo_index='geohash_4')
            self.union_geo_by_clients_4_df = pd.concat([self.union_geo_by_clients_4_df, geo_by_clients_4_df])
            self.clients_top_geohash_4_df = pd.concat([self.clients_top_geohash_4_df, pop_geo_by_clients_4_df])

            geo_by_clients_5_df, pop_geo_by_clients_5_df = calc_aggregate_client_by_geohash(geo_index='geohash_5')
            self.union_geo_by_clients_5_df = pd.concat([self.union_geo_by_clients_5_df, geo_by_clients_5_df])
            self.clients_top_geohash_5_df = pd.concat([self.clients_top_geohash_5_df, pop_geo_by_clients_5_df])

            geo_by_clients_6_df, pop_geo_by_clients_6_df = calc_aggregate_client_by_geohash(geo_index='geohash_6')
            self.union_geo_by_clients_6_df = pd.concat([self.union_geo_by_clients_6_df, geo_by_clients_6_df])
            self.clients_top_geohash_6_df = pd.concat([self.clients_top_geohash_6_df, pop_geo_by_clients_6_df])

            start_date = start_date + relativedelta(months=1)

    def save_output(self, path_output: str = 'datasets/'):
        # Сохроаняем: Частота преобретения клиентом продуктов пользующихся определеным банкоматом
        self.union_geohash_4_df.to_csv(path_output + 'union_geohash_4_df.csv', index=False)
        self.union_geohash_5_df.to_csv(path_output + 'union_geohash_5_df.csv', index=False)
        self.union_geohash_6_df.to_csv(path_output + 'union_geohash_6_df.csv', index=False)
        # Сохраняем: Связка клиента и используемые им банкоматы в отчетный период
        self.union_geo_by_clients_4_df.to_csv(path_output + 'union_geo_by_clients_4_df.csv', index=False)
        self.union_geo_by_clients_5_df.to_csv(path_output + 'union_geo_by_clients_5_df.csv', index=False)
        self.union_geo_by_clients_6_df.to_csv(path_output + 'union_geo_by_clients_6_df.csv', index=False)
        # Сохраняем: Частота преобретения клиентом продуктов пользующихся определеным банкоматом
        self.clients_top_geohash_4_df.to_csv(path_output + 'clients_top_geohash_4_df.csv', index=False)
        self.clients_top_geohash_5_df.to_csv(path_output + 'clients_top_geohash_5_df.csv', index=False)
        self.clients_top_geohash_6_df.to_csv(path_output + 'clients_top_geohash_6_df.csv', index=False)

if __name__ == '__main__':
    PATH = ''
    PATH_DATASET = PATH + 'datasets/sber_source/'
    PATH_DATASET_OUTPUT = PATH + 'datasets/'

    PATH_DATASET_GEO_TRAIN = PATH_DATASET + 'geo_train.parquet/'
    PATH_DATASET_GEO_TEST = PATH_DATASET + 'geo_test.parquet/'

    PATH_DATASET_TARGET_TRAIN = PATH_DATASET + 'train_target.parquet/'

    # Список ГЕО файлов для трейн и тест
    train_geo_files = glob.glob(PATH_DATASET_GEO_TRAIN + '/*.parquet')
    test_geo_files = glob.glob(PATH_DATASET_GEO_TEST + '/*.parquet')

    # Список файлов с реальными фактами продаж продуктов
    train_target_files = glob.glob(PATH_DATASET_TARGET_TRAIN + '/*.parquet')

    # Инициализируем класс. Загружаем первичные данные
    geo_features = GeoFeatures(train_target_files=train_target_files, train_geo_files=train_geo_files,
                               test_geo_files=test_geo_files)

    geo_features.calc_geohash_features()
    geo_features.calc_client_geo_features()

    geo_features.save_output()
