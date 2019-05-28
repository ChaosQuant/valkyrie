# coding=utf-8

import pdb
import datetime
import json
import pandas as pd
import numpy as np
import sqlalchemy as sa
from sqlalchemy import select, and_
from adjust_trade_date import AdjustTradeDate
from stock_pool import StockPool
from factors_engine import FactorsEngine
from alphamind.analysis.quantileanalysis import er_quantile_analysis
from alphamind.data.neutralize import neutralize
from alphamind.data.winsorize import winsorize_normal
from alphamind.data.standardize import standardize

class CalcFactorInfo(object):
    def __init__(self, file_setting='./factors.json'):
        __str__ = 'CalcFactorInfo'
        uqer_token = ''
        stock_conn = sa.create_engine('')
        self._stock_pool = StockPool(uqer_token=uqer_token, is_uqer=0)
        self._adjust_trade_date = AdjustTradeDate(uqer_token=uqer_token, is_uqer=1)
        self._factors_engine = FactorsEngine(file_setting)
        self._columns = []
        self._load_setting(file_setting)
        self._risk_columns = ['801010', '801020', '801030', '801040', '801050', '801080', '801110', '801120', '801130', 
                  '801140', '801150', '801160', '801170', '801180', '801200', '801210', '801230', '801710',
                  '801720', '801730', '801740', '801750', '801760', '801770', '801780', '801790', '801880','801890']
    
    def get_factor_columns(self):
        return self._columns
        
    def get_risk_columns(self):
        return self._risk_columns
    
    def _load_setting(self,file_setting):
        with open(file_setting) as f:
            self._factors_setting = json.load(f)
        for setting in self._factors_setting:
            factors_list = setting['factors']
            for factor in factors_list:
                self._columns += factor['columns']
    
#计算IC,IR,行业IR,五分位收益，行业IC半衰期    
    def calc_ic(self, factors,  windows):
        ic_dict = {}
        for column in self._columns:
            ic_series = factors.groupby('trade_date').apply(lambda x: np.corrcoef(x[column], 
                                                                          x['chgPct'])[0, 1]).fillna(0)
            ic_dict[column] = pd.DataFrame(ic_series, columns=['ic_values'])
        return ic_dict
    
    def calc_ir(self, factors, windows):
        ir_dict = {}
        for column in self._columns:
            ic_series = factors.groupby('trade_date').apply(lambda x: np.corrcoef(x[column], 
                                                                          x['chgPct'])[0, 1]).fillna(0)
            ir_dict[column] = (ic_series.mean()/ic_series.std())
        return ir_dict

    def calc_industry_ir(self, factors, windows):
        ir_industry_dict = {}
        for column in self._columns:
            industry_ic = factors.groupby(['trade_date','industryName']).apply(
                    lambda x: np.corrcoef(x[column], x['chgPct'])[0, 1]).fillna(0)
            industry_ir_series = (
                industry_ic.groupby(level=1).mean() / industry_ic.groupby(level=1).std()).fillna(0).sort_values(
                ascending=False)
            
            ir_industry_dict[column] = pd.DataFrame(industry_ir_series, columns=['ir_values'])
        return ir_industry_dict
    
    def calc_quantile(self, factors, bins = 5, de_trend = True):
        n_bins = 5
        quantile_dict = {}
        for column in self._columns:
            df = pd.DataFrame(columns=['q' + str(i) for i in range(1, n_bins+1)])
            grouped = factors.groupby('trade_date')
            for k, g in grouped:
                er = g[column].fillna(0).values
                dx_return = g['chgPct'].values
                res = er_quantile_analysis(er, n_bins=5, dx_return=dx_return, de_trend=de_trend)
                df.loc[k, :] = res
            quantile_dict[column] = df.reset_index().rename(columns={'index':'trade_date'}).set_index('trade_date')
        return quantile_dict
    
    def calc_return(self, factors):
        accum_dict = {}
        excess_dict = {}
        info_dict = {} #主动年化收益率 累计年化收益率
        #累计收益
        accum_return = self.calc_quantile(factors, bins = 5, de_trend = False)
        #超额收益
        excess_return = self.calc_quantile(factors, bins = 5, de_trend = True)
        for column in self._columns:
            accum_df = accum_return[column]
            #累计收益率换算
            if accum_df['q1'].cumsum()[-1] > accum_df['q5'].cumsum()[-1]:
                accum_df['calc_return'] = (accum_df['q1'] + 1).cumprod()
            else:
                accum_df['calc_return'] = (accum_df['q5'] + 1).cumprod()
            
            annualized_accum = (accum_df['calc_return'][-1]) ** (250/len(accum_df)) - 1
            accum_dict[column] = accum_df[['calc_return']]
            
            
            excess_df = excess_return[column]
            #累计收益率换算
            if excess_df['q1'].cumsum()[-1] > excess_df['q5'].cumsum()[-1]:
                excess_df['calc_return'] = (excess_df['q1'] + 1).cumprod()
            else:
                excess_df['calc_return'] = (excess_df['q5'] + 1).cumprod()
            
            annualized_excess = (excess_df['calc_return'][-1]) ** (250/len(excess_df)) - 1
            excess_dict[column] = excess_df[['calc_return']]
            
            
            info_dict[column] = {'accum':annualized_accum, 'excess':annualized_excess, 
                                 'IR':annualized_excess / excess_df['calc_return'].std()}
           
            
        return accum_dict,excess_dict,info_dict
    
    def calc_turnover_rate(self, factors):
        #获取五分位收益
        excess_return = self.calc_quantile(factors, bins = 5, de_trend = True)
        
        #获取股票池
      
    def calc_decay(self, factors, decay_interval=5):
        interval = decay_interval + 1
        decay_dict = {}
        for column in self._columns:
            factors_names = []
            factors_list = []
            values = {}
            grouped = factors.groupby(by='code')
            for k, group in grouped:
                group = group.sort_values(by='trade_date', ascending=True)
                for i in range(1, interval):
                    group[str(i) + '_' + column] = group[column].shift(0+i)
                factors_list += group[-interval:].to_dict(orient='records')
            new_factors_sets = pd.DataFrame(factors_list)
            for i in range(1, interval):
                factors_names.append(str(i) + '_' + column)
            industry_dummy = pd.get_dummies(new_factors_sets.indexSymbol)
            neutralized_factors = neutralize(industry_dummy.values.astype(float),
                                 new_factors_sets[factors_names].values,
                                 groups=new_factors_sets['trade_date'])
            new_factors_sets[factors_names] = neutralized_factors
            for f in factors_names:
                ic_series = new_factors_sets.groupby('trade_date').apply(lambda x: np.corrcoef(x[f].fillna(0), x['chgPct'])[0, 1])
                values[f] = ic_series.mean()
            values = pd.DataFrame([values])
            values.columns=['q' + str(i) for i in range(1, decay_interval+1)]
            decay_dict[column] = values
        return decay_dict
  
### 计算调仓日
    def adjustment_date(self, start_date, end_date, itype='isWeekEnd'):
        trade_date_list = self._adjust_trade_date.custom_fetch_end(start_date, end_date, itype)
        return trade_date_list
    
    def on_work_by_day(self, index, trade_date, stype, inveral = 10):
        print(index+':')
        columns = ['ROEAfterNonRecurring','CHV','IVR',
                  'EPAfterNonRecurring','DROEAfterNonRecurring',
                  'CFinc1','DRevenue']
        stock_pool = self._stock_pool.fetch_index(trade_date, index)
        factor_sets = self._raw_factor.on_work_by_day(trade_date, inveral, columns)
        factor_sets = factor_sets.merge(stock_pool[['code','weight','indexSymbol','industryName']], on=['code'])
        self.calc_ic(trade_date, factor_sets, columns, 1)
        self.calc_ir(trade_date, factor_sets, columns, 1)
        self.calc_industry_ir(trade_date, factor_sets, columns, 1)
        self.calc_quantile(factor_sets, columns)
        self.calc_decay(factor_sets, columns)
    
    def set_risk_style(self, risk_columns, is_reset =0):
        if is_reset == 1:
            self._risk_columns = ['801010', '801020', '801030', '801040', '801050', '801080', '801110', '801120', '801130', 
                  '801140', '801150', '801160', '801170', '801180', '801200', '801210', '801230', '801710',
                  '801720', '801730', '801740', '801750', '801760', '801770', '801780', '801790', '801880','801890']
            
        self._risk_columns += risk_columns
        self._risk_columns = list(set(self._risk_columns))
        
    def on_factors(self, index, trade_date_list):
        print(index+':')
        result_dict = {}
        factors_sets = self._factors_engine.on_work_by_interval(trade_date_list)
        stock_pool = self._stock_pool.on_work_by_interval(trade_date_list, 1, index)
        new_factors_sets = factors_sets.merge(stock_pool, on=['code','trade_date'])
        return new_factors_sets
    
    def on_factor_processing(self, new_factors_sets, columns = []):
        calc_columns = columns if len(columns) > 0 else self._columns
        ### 根据因子种类不同，做nan处理，基本面因子(成长，价值，质量)采用行业中值处理，其他以0处理,
        #暂时以0处理
        for column in calc_columns:
            new_factors_sets[column] = new_factors_sets[column].fillna(0)
        #去极值
        for column in  calc_columns:
            new_factors_sets['winsorize_' + column] = winsorize_normal(new_factors_sets[column].values.reshape(-1,1),
                                                                       num_stds=1).flatten()
        #行业风险中性化
        for column in  calc_columns:
            new_factors_sets['neutralize_' + column] = neutralize(
                new_factors_sets[self._risk_columns].values.astype(float), 
                new_factors_sets['winsorize_' + column].values).flatten()
        
        #标准化
        for column in calc_columns:
            new_factors_sets['standardize_' + column] = standardize(
                new_factors_sets['neutralize_' + column].values.reshape(-1,1))
        #暂时以0处理
        for column in calc_columns:
            new_factors_sets[column] = new_factors_sets[column].fillna(0)
        return new_factors_sets
        
    def on_factors_by_trade_list(self, index, start_date, end_date, columns):
        new_factors_sets =  self.on_factors(index, trade_date_list, columns)
        new_factors_sets = new_factors_sets.fillna(0)
        new_factors_sets = self.on_factor_processing(new_factors_sets)
        return new_factors_sets
        
    def on_work_by_trade_list(self, index, trade_date_list, bins, decay_interval):
        result_dict = {}
        new_factors_sets = self.on_factors(index, trade_date_list)
        new_factors_sets = self.on_factor_processing(new_factors_sets)
        #IC序列
        result_dict['ic_serialize'] = self.calc_ic(new_factors_sets, 0)
        #IR
        result_dict['ir_value'] = self.calc_ir(new_factors_sets, 0)
        #行业IR
        result_dict['industry_ir'] = self.calc_industry_ir(new_factors_sets, 0)
        #分位收益
        result_dict['quantile'] = self.calc_quantile(new_factors_sets, bins=bins, de_trend=True)
        #半衰周期
        result_dict['decay'] = self.calc_decay(new_factors_sets, decay_interval=decay_interval)
        #因子累计收益和超额收益计算 用5分位中表现最佳一组 计算年化收益率 主动年化收益率
        result_dict['return'] = self.calc_return(new_factors_sets)
        # 换手率
        
        return result_dict
    
        
    def on_work_by_week(self, index, start_date, end_date):
        pdb.set_trace()
        #获取调仓时间周期
        trade_date_list = self._adjust_trade_date.custom_fetch_end(start_date, end_date, 'isWeekEnd')
        #为了防止乱序，多加一次时间列表排序
        trade_date_list.sort(reverse=False)
        result_dict =  self.on_work_by_trade_list(index, trade_date_list, 5, 5)
        return result_dict
        
     
if __name__ == '__main__':
    start_date = datetime.datetime(2018, 1, 1).date()
    end_date = datetime.datetime(2018,12,31).date()
    calc_factor_info = CalcFactorInfo()
    calc_factor_info.set_risk_style(['SIZE'])
    new_factors_sets = calc_factor_info.on_work_by_week('000016.XSHG', start_date, end_date)
    print(new_factors_sets)

