import pandas as pd

# 获取给定日期前或后n天的所有交易日期
def Get_All_Tradingday(engine, date:str, n:int, forward=False)->list:
    """
    功能: 获取给定日期前或后n天的所有交易日期
    输入: 
        engine: 默认为"create_engine('mssql+pyodbc://visitor:123@192.168.5.18/daily?driver=SQL+Server')"
        date: 字符串,代表日期,如'20210101'
        n: int,代表date日期前或后所需要的交易日期天数
        forward: 默认为False,代表日期往后
    输出: 
        [time_1,time_2,.....],其中time_i为Timestamp日期格式

    Example:
    >>> Get_All_Tradingday(engine,date='20220101',n=3)

    返回: [Timestamp('2022-01-04 00:00:00'),Timestamp('2022-01-05 00:00:00'),Timestamp('2022-01-06 00:00:00')]
    """

    if forward:
        query_pre_end_tradingday = f"""SELECT TOP {n} * FROM daily..tradingday WHERE tradingday < '{date}' ORDER BY tradingday DESC"""
        start_date = pd.read_sql(query_pre_end_tradingday, engine).tradingday.tolist()[-1]
        query_trad_tradingday = f"SELECT distinct tradingday FROM daily..tradingday WHERE  tradingday >= '{start_date}' AND tradingday <= '{date}'"  
        tradingday_list = sorted(list(set(pd.to_datetime(pd.read_sql(query_trad_tradingday, engine).tradingday).tolist())))
    else:
        query_pre_end_tradingday = f"""SELECT TOP {n} * FROM daily..tradingday WHERE tradingday > '{date}' ORDER BY tradingday"""
        end_date = pd.read_sql(query_pre_end_tradingday, engine).tradingday.tolist()[-1]
        query_trad_tradingday = f"SELECT distinct tradingday FROM daily..tradingday WHERE  tradingday >= '{date}' AND tradingday <= '{end_date}'"  
        tradingday_list = sorted(list(set(pd.to_datetime(pd.read_sql(query_trad_tradingday, engine).tradingday).tolist())))
    return tradingday_list

# 获取起始日期到终止日期的各行业名称
def get_stock_industry(engine, beginday:str, endday:str)->list:
    """
    功能: 获取起始日期到终止日期的各行业名称
    输入: 
        engine: 默认为"create_engine('mssql+pyodbc://visitor:123@192.168.5.18/daily?driver=SQL+Server')"
        start: 字符串,如'20210101'
        end: 字符串,如'20220101'
    输出:
        ['交运设备','交通运输',......]

    Example:
    >>> get_stock_industry(engine,beginday='20220101',endday='20230101')

    返回: [Timestamp('2022-01-04 00:00:00'),Timestamp('2022-01-05 00:00:00'),Timestamp('2022-01-06 00:00:00')]
    """
    query_stock_df = f'''WITH raw AS (
                        SELECT a.tradingday,a.code,[open],high,low,[close],pre_close,volume,turnover,b.industry,b.totalMV,b.negotiableMV,
                        volume_chg=CASE WHEN lag(volume,1,NULL) OVER (ORDER BY a.code,a.tradingday) <= 0 THEN 1 ELSE volume*1.0/lag(volume,1,NULL) OVER (ORDER BY a.code,a.tradingday) END,
                        turnover_chg=CASE WHEN lag(turnover,1,NULL) OVER (ORDER BY a.code,a.tradingday) <= 0 THEN 1 ELSE turnover*1.0/lag(turnover,1,NULL) OVER (ORDER BY a.code,a.tradingday) END,
                        totalMV_chg=CASE WHEN lag(totalMV,1,NULL) OVER (ORDER BY a.code,a.tradingday) <= 0 THEN 1 ELSE totalMV*1.0/lag(totalMV,1,NULL) OVER (ORDER BY a.code,a.tradingday) END,
                        negotiableMV_chg=CASE WHEN lag(negotiableMV,1,NULL) OVER (ORDER BY a.code,a.tradingday) <= 0 THEN 1 ELSE negotiableMV*1.0/lag(negotiableMV,1,NULL) OVER (ORDER BY a.code,a.tradingday) END
                        FROM daily..daybar a 
                        JOIN daily..stockinfo b ON a.tradingday = b.tradingday AND a.code = b.code
                        WHERE a.code IN (
                        SELECT code FROM (SELECT tradingday,code,tradeable=CASE WHEN datediff(day,listeddate,tradingday) > 180 THEN 1 ELSE 0 END&tradeable 
                        FROM daily..stockinfo WHERE tradingday BETWEEN '{beginday}' AND '{endday}') x GROUP BY code HAVING count(*) = sum(tradeable))	-- 剔除次新股和ST股票
                        AND a.tradingday BETWEEN dateadd(month,-1,'{beginday}') AND '{endday}'
                        )
                        SELECT a.tradingday,a.code,

                        a.industry
                        FROM raw a
                        LEFT JOIN (
                            SELECT tradingday,industry,industry_open=avg([open]),industry_high=avg(high),industry_low=avg(low),industry_close=avg([close]) FROM raw GROUP BY tradingday,industry
                        ) b ON a.tradingday = b.tradingday AND a.industry = b.industry
                        WHERE a.tradingday BETWEEN '{beginday}' AND '{endday}'
                        ORDER BY a.code,a.tradingday'''
    df = pd.read_sql(query_stock_df, engine)
    industry_list = sorted(list(set(df.industry.tolist())))
    return industry_list

# 获取行业内所有股票
def get_industry_stock(engine, beginday, endday, industry):
    """
    功能: 获取起始日期到终止日期的某行业的所有股票
    输入: 
        engine: 默认为"create_engine('mssql+pyodbc://visitor:123@192.168.5.18/daily?driver=SQL+Server')"
        start: 字符串,如'20210101'
        end: 字符串,如'20220101'
        industry: 字符串,如'交通运输'
    输出:
        ['交运设备','交通运输',......]
    
    Example:
    >>> get_industry_stock(engine,beginday='20220101',endday='20230101',industry='交运设备')

    返回: ['SH600862', 'SH600967', 'SZ000519', 'SZ002190']
    """
    query_stock_df = f'''WITH raw AS (
                        SELECT a.tradingday,a.code,[open],high,low,[close],pre_close,volume,turnover,b.industry,b.totalMV,b.negotiableMV,
                        volume_chg=CASE WHEN lag(volume,1,NULL) OVER (ORDER BY a.code,a.tradingday) <= 0 THEN 1 ELSE volume*1.0/lag(volume,1,NULL) OVER (ORDER BY a.code,a.tradingday) END,
                        turnover_chg=CASE WHEN lag(turnover,1,NULL) OVER (ORDER BY a.code,a.tradingday) <= 0 THEN 1 ELSE turnover*1.0/lag(turnover,1,NULL) OVER (ORDER BY a.code,a.tradingday) END,
                        totalMV_chg=CASE WHEN lag(totalMV,1,NULL) OVER (ORDER BY a.code,a.tradingday) <= 0 THEN 1 ELSE totalMV*1.0/lag(totalMV,1,NULL) OVER (ORDER BY a.code,a.tradingday) END,
                        negotiableMV_chg=CASE WHEN lag(negotiableMV,1,NULL) OVER (ORDER BY a.code,a.tradingday) <= 0 THEN 1 ELSE negotiableMV*1.0/lag(negotiableMV,1,NULL) OVER (ORDER BY a.code,a.tradingday) END
                        FROM daily..daybar a 
                        JOIN daily..stockinfo b ON a.tradingday = b.tradingday AND a.code = b.code
                        WHERE a.code IN (
                        SELECT code FROM (SELECT tradingday,code,tradeable=CASE WHEN datediff(day,listeddate,tradingday) > 180 THEN 1 ELSE 0 END&tradeable 
                        FROM daily..stockinfo WHERE tradingday BETWEEN '{beginday}' AND '{endday}') x GROUP BY code HAVING count(*) = sum(tradeable))	-- 剔除次新股和ST股票
                        AND a.tradingday BETWEEN dateadd(month,-1,'{beginday}') AND '{endday}'    
                        )
                        SELECT a.tradingday,a.code,

                        a.industry
                        FROM raw a
                        LEFT JOIN (
                            SELECT tradingday,industry,industry_open=avg([open]),industry_high=avg(high),industry_low=avg(low),industry_close=avg([close]) FROM raw GROUP BY tradingday,industry
                        ) b ON a.tradingday = b.tradingday AND a.industry = b.industry
                        WHERE a.tradingday BETWEEN '{beginday}' AND '{endday}'
                        ORDER BY a.code,a.tradingday'''
    df = pd.read_sql(query_stock_df, engine)
    stock_list = sorted(list(set(df[df['industry'] == industry].code.tolist())))
    return stock_list

# 获得单个股票信息
def get_stock(engine, beginday, endday, stock_code:str):
    '''
    功能: 获取起始日期到终止日期的单只股票信息
    输入: 
        engine: 默认为"create_engine('mssql+pyodbc://visitor:123@192.168.5.18/daily?driver=SQL+Server')"
        start: 字符串,如'20210101'
        end: 字符串,如'20220101'
        industry: 字符串,如'SH600862'
    输出:
        ['交运设备','交通运输',......]
    
    Example:
    >>> get_stock(engine,beginday='20220101',endday='20230101',stock_code='SH600862')

    返回: 一个DataFrame
    '''
    query_stock_df = f'''SELECT * FROM daybar WHERE code IN ('{stock_code}') AND tradingday BETWEEN {beginday} AND {endday}'''
    df = pd.read_sql(query_stock_df, engine)
    df.tradingday = pd.to_datetime(df.tradingday)
    df = df.set_index(['code'])
    df.sort_index(inplace=True)
    return_ = ((df['close'] / df['pre_close'])-1)
    df['return'] = return_
    df.drop(['pre_close', 'factor', 'turnover'], axis=1, inplace=True)
    df.dropna(inplace=True)
    df.rename(columns={'tradingday': 'date'}, inplace=True)
    return df

# 获取某个行业的股票信息(纵向排列)
def get_stock_info(engine, beginday, endday, industry:str):
    """
    功能: 获取起始日期到终止日期的某个行业的股票信息
    输入: 
        engine: 默认为"create_engine('mssql+pyodbc://visitor:123@192.168.5.18/daily?driver=SQL+Server')"
        start: 字符串,如'20210101'
        end: 字符串,如'20220101'
        industry: 字符串,如'交通运输'
    输出:
        ['交运设备','交通运输',......]
    
    Example:
    >>> get_stock_info(engine,beginday='20220101',endday='20230101',industry='交通运输')

    返回: 一个DataFrame
    """
    query_stock_df = f'''WITH raw AS (
                        SELECT a.tradingday,a.code,[open],high,low,[close],pre_close,volume,turnover,b.industry,b.totalMV,b.negotiableMV,
                        volume_chg=CASE WHEN lag(volume,1,NULL) OVER (ORDER BY a.code,a.tradingday) <= 0 THEN 1 ELSE volume*1.0/lag(volume,1,NULL) OVER (ORDER BY a.code,a.tradingday) END,
                        turnover_chg=CASE WHEN lag(turnover,1,NULL) OVER (ORDER BY a.code,a.tradingday) <= 0 THEN 1 ELSE turnover*1.0/lag(turnover,1,NULL) OVER (ORDER BY a.code,a.tradingday) END,
                        totalMV_chg=CASE WHEN lag(totalMV,1,NULL) OVER (ORDER BY a.code,a.tradingday) <= 0 THEN 1 ELSE totalMV*1.0/lag(totalMV,1,NULL) OVER (ORDER BY a.code,a.tradingday) END,
                        negotiableMV_chg=CASE WHEN lag(negotiableMV,1,NULL) OVER (ORDER BY a.code,a.tradingday) <= 0 THEN 1 ELSE negotiableMV*1.0/lag(negotiableMV,1,NULL) OVER (ORDER BY a.code,a.tradingday) END
                        FROM daily..daybar a 
                        JOIN daily..stockinfo b ON a.tradingday = b.tradingday AND a.code = b.code
                        WHERE a.code IN (
                        SELECT code FROM (SELECT tradingday,code,tradeable=CASE WHEN datediff(day,listeddate,tradingday) > 180 THEN 1 ELSE 0 END&tradeable 
                        FROM daily..stockinfo WHERE tradingday BETWEEN '{beginday}' AND '{endday}') x GROUP BY code HAVING count(*) = sum(tradeable))	-- 剔除次新股和ST股票
                        AND a.tradingday BETWEEN dateadd(month,-1,'{beginday}') AND '{endday}'
                        )
                        SELECT a.tradingday,a.code,

                        ------价格四选一----
                        -- 原始价格
                        [open],high,low,[close],pre_close,
                        /*
                        --价格转比例
                        [open]=[open]/pre_close,high=[high]/pre_close,low=[low]/pre_close,[close]=[close]/pre_close,

                        --原始价格去极值
                        [open]=CASE WHEN [open]/pre_close > 1.1 THEN pre_close * 1.1 WHEN [open]/pre_close < 0.9 THEN pre_close * 0.9 ELSE [open] END,
                        high=CASE WHEN [high]/pre_close > 1.1 THEN pre_close * 1.1 WHEN [high]/pre_close < 0.9 THEN pre_close * 0.9 ELSE [high] END,
                        low=CASE WHEN [low]/pre_close > 1.1 THEN pre_close * 1.1 WHEN [low]/pre_close < 0.9 THEN pre_close * 0.9 ELSE [low] END,
                        [close]=CASE WHEN [close]/pre_close > 1.1 THEN pre_close * 1.1 WHEN [close]/pre_close < 0.9 THEN pre_close * 0.9 ELSE [close] END,


                        --价格转比例去极值
                        [open]=CASE WHEN [open]/pre_close > 1.1 THEN pre_close * 1.1 WHEN [open]/pre_close < 0.9 THEN pre_close * 0.9 ELSE [open]/pre_close END,
                        high=CASE WHEN [high]/pre_close > 1.1 THEN pre_close * 1.1 WHEN [high]/pre_close < 0.9 THEN pre_close * 0.9 ELSE [high]/pre_close END,
                        low=CASE WHEN [low]/pre_close > 1.1 THEN pre_close * 1.1 WHEN [low]/pre_close < 0.9 THEN pre_close * 0.9 ELSE [low]/pre_close END,
                        [close]=CASE WHEN [close]/pre_close > 1.1 THEN pre_close * 1.1 WHEN [close]/pre_close < 0.9 THEN pre_close * 0.9 ELSE [close]/pre_close END,
                        -------end------
                        */
                        ------量额四选一----
                        --原始量额
                        volume,
                        /*
                        --量额转比例
                        volume=volume_chg,
                        turnover=turnover_chg,

                        --原始量额去极值
                        volume = CASE WHEN volume_chg = 0 THEN 0 WHEN volume_chg > 10 THEN volume/volume_chg*10 WHEN volume_chg < 0.1 THEN volume/volume_chg*0.1 ELSE volume END,
                        turnover = CASE WHEN turnover_chg = 0 THEN 0 WHEN turnover_chg > 10 THEN turnover/turnover_chg*10 WHEN turnover_chg < 0.1 THEN turnover/turnover_chg*0.1 ELSE turnover END,

                        --量额转比例去极值
                        volume = CASE WHEN volume_chg = 0 THEN 1 WHEN volume_chg > 10 THEN 10 WHEN volume_chg < 0.1 THEN 0.1 ELSE volume_chg END,
                        turnover = CASE WHEN turnover_chg = 0 THEN 1 WHEN turnover_chg > 10 THEN 10 WHEN turnover_chg < 0.1 THEN 0.1 ELSE turnover_chg END,
                        -------end------
                        */
                        a.industry
                        FROM raw a
                        LEFT JOIN (
                            SELECT tradingday,industry,industry_open=avg([open]),industry_high=avg(high),industry_low=avg(low),industry_close=avg([close]) FROM raw GROUP BY tradingday,industry
                        ) b ON a.tradingday = b.tradingday AND a.industry = b.industry
                        WHERE a.tradingday BETWEEN '{beginday}' AND '{endday}'
                        ORDER BY a.code,a.tradingday'''
    df = pd.read_sql(query_stock_df, engine)
    df.tradingday = pd.to_datetime(df.tradingday)
    # df = df.set_index(['code'])
    df = df.set_index(['code','tradingday'])
    df.sort_index(inplace=True)
    if industry is not None:
        df = df[df['industry'] == industry]
    return_ = (((df['close'] / df['pre_close'])-1))
    df['return'] = return_
    df.drop(['pre_close', 'industry'], axis=1, inplace=True)
    df.dropna(inplace=True)
    df.rename(columns={'tradingday': 'date'}, inplace=True)
    return df


def Get_data(engine, end, train_len, iag_len,industry=None,stock=None):
    '''
    一次性批量读取数据。为保证每只股票有足够的天数，多取iag_len 天，后切分.
    '''
    require_start_tradingday = f'''SELECT TOP {train_len+iag_len} * FROM daily..tradingday WHERE tradingday < '{end}' ORDER BY tradingday DESC'''
    start = pd.read_sql(require_start_tradingday, engine).tradingday.tolist()[-1]
    if industry is not None:
        data = get_stock_info(engine, start, end, industry)
    elif stock is not None:
        data = get_stock(engine, start, end, stock)
    elif industry is None and stock is None:
        data = get_stock_info(engine, start, end, industry)
    if data.empty:
        raise ValueError('取数出错,DataFrame为空')
    
    return data