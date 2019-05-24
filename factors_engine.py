# coding=utf-8

import pdb
import json
import datetime
import importlib
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import select, and_
from models import Market,RiskExposure
from raw_factors import RawFactors
from adjust_trade_date import AdjustTradeDate
import config

class FactorsEngine(object):
    
    def __init__(self, file_setting='./factors.json'):
        self._file_setting = file_setting
        self._load_setting()
        self._return_conn = sa.create_engine(config.quant_db)
        self._risk_conn = sa.create_engine(config.quant_db)
   
    def _load_setting(self):
        with open(self._file_setting) as f:
            self._factors_setting = json.load(f)
    
    def on_return_by_interval(self, trade_date_list):
        query = select([Market.trade_date, Market.code,Market.closePrice]).where(
            and_(
                Market.trade_date.in_(trade_date_list)
            )
        ).order_by(Market.trade_date)
        df = pd.read_sql(query, con=self._return_conn)
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df.sort_values(by='trade_date')
        return df
    
    def on_risk_by_interval(self, trade_date_list):
        query = select([RiskExposure]).where(
            and_(
                RiskExposure.trade_date.in_(trade_date_list)
            )
        ).order_by(RiskExposure.trade_date)
        df = pd.read_sql(query, con=self._risk_conn)
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df.sort_values(by='trade_date')
        return df
        
    def on_work_by_factors(self, trade_date_list):
        factors_sets = None
        for setting in self._factors_setting:
            conn = setting['conn']
            factors_list = setting['factors']
            for factor in factors_list:
                raw_factors = RawFactors(conn)
                model_name = factor['model']
                table_name = factor['table']
                code_name = factor['code']
                trade_date_name = factor['trade_date']
                columns =  factor['columns']
                module = importlib.import_module(model_name)
                table = module.__getattribute__(table_name)
                factor_sets = raw_factors.on_work_by_interval(trade_date_list, table, code_name, trade_date_name,
                                                              columns)
                if code_name != 'code':
                    factor_sets = factor_sets.rename(columns={'symbol':'code'})
                if factors_sets is None:
                    factors_sets = factor_sets
                else:
                    factors_sets = factors_sets.merge(factor_sets, on=['code','trade_date'])
        return factors_sets
    
    def on_main_factors(self, trade_date_list, factor_sets, return_sets):
        #以因子日期为主，即把T+1的收益放在T期
        trade_date_list.sort(reverse=False)
        grouped = return_sets.groupby(by='trade_date')
        new_return_list = []
        for k, group in grouped:
            index = trade_date_list.index(datetime.datetime.strptime(k.strftime('%Y-%m-%d'),'%Y-%m-%d').date())
            group.loc[:,'trade_date'] = trade_date_list[index-1]
            new_return_list += group.to_dict(orient='records')
        new_return_sets = pd.DataFrame(new_return_list)
        new_return_sets['trade_date'] = pd.to_datetime(new_return_sets['trade_date'])
        return factor_sets.merge(new_return_sets, on=['code', 'trade_date'])
        
    def on_main_return(self, trade_date_list, factors, return_sets):
        #以收益率日期为主,即把T期因子放在T+1期
        trade_date_list.sort(reverse=False)
        grouped = factors.groupby(by='trade_date')
        factors_list = []
        for k, group in grouped:
            index = trade_date_list.index(datetime.datetime.strptime(k.strftime('%Y-%m-%d'),'%Y-%m-%d').date())
            group['trade_date'] = trade_date_list[index-1]
            factors_list += group.to_dict(orient='records')
        new_factors_sets = pd.DataFrame(factors_list)
        return new_factors_sets.merge(return_sets, on=['code', 'trade_date'])
    
    def on_work_by_interval(self, trade_date_list, main_type=1):
        trade_date_list.sort(reverse=False)
        factors_sets = self.on_work_by_factors(trade_date_list[:-1])
        #获取当期收益率
        now_return_sets = self.on_return_by_interval(trade_date_list[:-1])
        factors_sets = factors_sets.merge(now_return_sets,on=['code','trade_date']).rename(columns={'closePrice':'nclosePrice'})
        return_sets = self.on_return_by_interval(trade_date_list[1:])
        if main_type == 1: # 当前日期为T
            factors_sets = self.on_main_factors(trade_date_list, factors_sets, return_sets)
        else: # 当前日期为T+1
            factors_sets = self.on_main_return(trade_date_list, factors_sets, return_sets)
        factors_sets['chgPct']  = (factors_sets['closePrice'] / factors_sets['nclosePrice']) -1
        #加入风险因子
        risk_sets = self.on_risk_by_interval(trade_date_list)
        risk_sets.rename(columns={'Bank':'801780','RealEstate':'801180','Health':'801150',
                                 'Transportation':'801170','Mining':'801020','NonFerMetal':'801050',
                                 'HouseApp':'801110','LeiService':'801210','MachiEquip':'801890',
                                 'BuildDeco':'801720','CommeTrade':'801200','CONMAT':'801710',
                                 'Auto':'801880','Textile':'801130','FoodBever':'801120',
                                 'Electronics':'801080','Computer':'801750','LightIndus':'801140',
                                 'Utilities':'801160','Telecom':'801770','AgriForest':'801010',
                                 'CHEM':'801030','Media':'801760','IronSteel':'801040','NonBankFinan':'801790',
                                 'ELECEQP':'801730','AERODEF':'801740','Conglomerates':'801230'}, inplace=True)
        factors_sets = factors_sets.merge(risk_sets, on=['code','trade_date'])
        return factors_sets

if __name__ == "__main__":
    factors_engine = FactorsEngine()
    uqer_token = config.uqer_token
    adjust_trade = AdjustTradeDate(uqer_token = uqer_token)
    start_date = datetime.datetime(2018, 5, 29).date()
    end_date = datetime.datetime(2018, 7, 9).date()
    trade_date_list = adjust_trade.custom_fetch_end(start_date, end_date, 'isWeekEnd')
    factors_engine.on_work_by_interval(trade_date_list)