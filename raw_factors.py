# coding=utf-8
import pdb
import datetime
import pandas as pd
import numpy as np
from sqlalchemy import select, and_
import sqlalchemy as sa


class RawFactors(object):
    def __init__(self, conn):
        self._factors_conn = conn
    
    def custom_factors_by_interval(self, trade_date_list, table, code_name, trade_name, columns):
        db_columns = []
        if len(columns) == 0:
            db_columns.append(table)
        else:
            db_columns.append(table.__dict__[code_name])
            db_columns.append(table.__dict__[trade_name])
            for column in columns:
                db_columns.append(table.__dict__[column])
        query = select(db_columns).where(
            and_(
                table.trade_date.in_(trade_date_list)
            )
        ).order_by(table.trade_date)
        df = pd.read_sql(query, con=self._factors_conn)
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        return df
        
    def on_work_by_interval(self, trade_date_list, table, code_name, trade_name, columns=[]):
        trade_date_list.sort(reverse=False)
        factor_sets = self.custom_factors_by_interval(trade_date_list[:-1], table, code_name,
                                                      trade_name, columns)
        return factor_sets
    
        
if __name__ == '__main__':
    uqer_token = ''
    raw_factor = RawFactors()
    adjust_trade = AdjustTradeDate(uqer_token = uqer_token)
    start_date = datetime.datetime(2018, 5, 29).date()
    end_date = datetime.datetime(2018, 7, 9).date()
