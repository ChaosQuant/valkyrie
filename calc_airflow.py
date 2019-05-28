# coding=utf-8

import os
import pdb
import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
import datetime
import config
from calc_factor_info import CalcFactorInfo


class CalcAirFlow(object):
    def __init__(self):
        #目标数据库
        self._destination = sa.create_engine(config.vision_db)
        #self._destination = sa.create_engine("")
        self._destsession = sessionmaker( bind=self._destination, autocommit=False, autoflush=True)
        self._calc_factor_info = CalcFactorInfo()
    
    def update_destdb(self, table_name, code, factor, time_type, sets, end_date = None, is_reset = True):
        sets = sets.where(pd.notnull(sets), None)
        if is_reset:
            sets = sets.reset_index()
        sets['factor'] = factor
        sets['code'] = code
        sets['time_type'] = time_type
        if end_date is not None:
            sets['trade_date'] = end_date
            sets['trade_date'] = pd.to_datetime(end_date)
        sql_pe = 'INSERT INTO {0} SET'.format(table_name)
        updates = ",".join( "{0} = :{0}".format(x) for x in list(sets) )
        sql_pe = sql_pe + '\n' + updates
        sql_pe = sql_pe + '\n' +  'ON DUPLICATE KEY UPDATE'
        sql_pe = sql_pe + '\n' + updates
        session = self._destsession()
        print('update_destdb')
        for index, row in sets.iterrows():
            dict_input = dict( row )
            dict_input['trade_date'] = dict_input['trade_date'].to_pydatetime()
            session.execute(sql_pe, dict_input)
        session.commit()
        session.close()
        
    def _create_trade_date(self, start_date, end_date, time_type):
        return self._calc_factor_info.adjustment_date(start_date, end_date, time_type)
    
    def on_work_by_week(self, index, start_date, end_date):
        trade_date_list = self._create_trade_date(self, start_date, end_date, 'isWeekEnd')
        result_dict = self._calc_factor_info.on_factors(index,trade_date_list)
        return result_dict
    
    def on_work_sets_by_interval(self, index_sets, trade_date_list, interval):
        for index in index_sets:
            result_dict = self._calc_factor_info.on_work_by_trade_list(index,trade_date_list, 5, 5)
            columns = list(result_dict.keys())
            factor_columns = self._calc_factor_info.get_factor_columns()
            for factor in factor_columns:
                self.update_destdb('ic_serialize', index, factor, interval, result_dict['ic_serialize'][factor])
                self.update_destdb('industry_ir', index, factor, interval, result_dict['industry_ir'][factor], end_date)
                self.update_destdb('quantile', index, factor, interval, result_dict['quantile'][factor])
                self.update_destdb('ic_decay', index, factor, interval, result_dict['decay'][factor], end_date, False)
            print('isWeekEnd',index)
    
    def on_work_sets_by_day(self, index_sets, start_date, end_date):
        trade_date_list = self._create_trade_date(start_date, end_date, 'isOpen')
        self.on_work_sets_by_interval(index_sets, trade_date_list, 0)
        
    def on_work_sets_by_week(self, index_sets, start_date, end_date):
        trade_date_list = self._create_trade_date(start_date, end_date, 'isWeekEnd')
        self.on_work_sets_by_interval(index_sets, trade_date_list, 1)
        
    def on_work_sets_by_month(self, index_sets, start_date, end_date):
        trade_date_list = self._create_trade_date(start_date, end_date, 'isMonthEnd')
        self.on_work_sets_by_interval(index_sets, trade_date_list, 2)
    
    
if __name__ == '__main__':
    calc_air_flow = CalcAirFlow()
    start_date = datetime.datetime(2017, 1, 1).date()
    end_date = datetime.datetime(2018,12,31).date()
    index_sets = ['000905.XSHG','000300.XSHG','000906.XSHG']
    
    calc_air_flow.on_work_sets_by_day(index_sets, datetime.datetime(2018, 10, 1).date(), 
                                       datetime.datetime(2018,12,31).date())
                                      
    calc_air_flow.on_work_sets_by_week(index_sets, datetime.datetime(2017, 1, 1).date(), 
                                       datetime.datetime(2018,12,31).date())
    
    calc_air_flow.on_work_sets_by_month(index_sets, datetime.datetime(2010, 1, 1).date(), 
                                       datetime.datetime(2018,12,31).date())