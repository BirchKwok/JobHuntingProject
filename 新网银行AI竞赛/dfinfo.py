import pandas as pd
import numpy as np


class DataFrameInfo(pd.DataFrame):
    '''
    定义一个类，继承pandas.DataFrame，方便查看数据和其他EDA定制化数据查看需求。
    '''

    def preview(self):
        '''
        用于数据预览，查看数据集的各种数据属性。


        returns:

        total: 数量,
        na: 空值,
        naPercent: 空值所占该列比例,
        dtype: 该列的数据类型(dtype),
        max: 该列最大值,
        75%: 75%分位数,
        median: 中位数,
        25%: 25%分位数,
        min:该列最小值,
        mean: 均值,
        mode: 众数,
        variation: 异众比率,
        std: 标准差,
        skew: 偏度系数,
        kurt: 峰度系数,
        sampleVals: 随机返回该列两个值
        '''
        ind = self.index
        col = self.columns
        ind_len = self.shape[0]
        col_len = len(col)
        df = pd.DataFrame(columns=['total', 'na', 'naPercent', 'dtype', 'max', '75%', 'median',
                                   '25%', 'min', 'mean', 'mode', 'variation', 'std', 'skew', 'kurt', 'sampleVals'])

        pointer = 0
        for i in col:
            sampleVals = ' || '.join([str(self[i][j]) for j in np.random.randint(ind_len, size=2)])
            if sum(self.iloc[:, pointer].apply(lambda s: isinstance(s, (float, int)))) == \
                    len(self.iloc[:, pointer]):
                value = {'total': ind_len,
                         'na': self[i].isnull().sum(),
                         'naPercent': self[i].isnull().sum() / ind_len,
                         'dtype': self[i].dtype,
                         'max': self[i].max(),
                         '75%': self[i].quantile(0.75),
                         'median': self[i].median(),
                         '25%': self[i].quantile(0.25),
                         'min': self[i].min(),
                         'mean': self[i].mean(),
                         'mode': self[i].value_counts(ascending=False).index[0],
                         'variation': self[i].value_counts(ascending=False).values[1:].sum() / ind_len,
                         'std': self[i].std(),
                         'skew': self[i].skew(),
                         'kurt': self[i].kurt(),
                         'sampleVals': sampleVals
                         }
            else:
                value = {'total': ind_len,
                         'na': self[i].isnull().sum(),
                         'naPercent': self[i].isnull().sum() / ind_len,
                         'dtype': self[i].dtype,
                         'max': np.nan,
                         '75%': np.nan,
                         'median': np.nan,
                         '25%': np.nan,
                         'min': np.nan,
                         'mean': np.nan,
                         'mode': self[i].value_counts(ascending=False).index[0],
                         'variation': self[i].value_counts(ascending=False).values[1:].sum() / ind_len,
                         'std': np.nan,
                         'skew': np.nan,
                         'kurt': np.nan,
                         'sampleVals': sampleVals
                         }
            df.loc[i] = pd.Series(value, name=i)
            pointer += 1

        return df

    def _countNullStringAndGetType(self, col):
        '''
        判断传入列的每个值是否为空字符串，并获取该值数据类型。

        params:
        col: pandas Series

        returns：

        一个列表，包含该列的空字符串数目、以及一个该列所有值的数据类型的set集合。
        '''
        count = 0
        types = set()
        for j in col:
            if isinstance(j, str) and j.strip() == "":
                count += 1
            types.add(type(j))

        return [count, types]

    def abnormal(self):
        '''
        查看数据集的异常值情况。


        returns：

        na: 该列的空值数目,
        nullStrings: 该列空字符串数目,
        valueTypes：该列所有数值的数据类型set集合
        '''

        df = pd.DataFrame(columns=['na', 'nullStrings', 'valueTypes'])

        for i in self.columns:
            df.loc[i] = pd.Series({
                'na': sum(self[i].isna()),
                'nullStrings': self._countNullStringAndGetType(self[i])[0],
                'valueTypes': self._countNullStringAndGetType(self[i])[1]
            }, name=i)


        return df