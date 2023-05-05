# coding: utf-8
import lightgbm as lgb
import pandas as pd
import os
import pickle

from lightgbm_ import GBDT
from mlpregressor_ import MLPR
from linearregression_ import LinearR

class stacking():
    fdir = os.path.dirname(__file__)
    source_dir = os.path.join(fdir, '../sources/processed')

    def __init__(self, desc):
        self.desc = desc
        # super().__init__(desc)

    def getdatasets(self):
        self.tr = pickle.load(open(os.path.join(self.source_dir, 'nan_train.pkl'), 'rb'))
        self.te = pickle.load(open(os.path.join(self.source_dir, 'nan_test.pkl'), 'rb'))
    
    def splity(self, df):
        x = df.drop(columns=['cover'])
        y = df['cover']
        return x, y

    def extenddata(self, cl):
        ins = cl(self.desc)
        ins.desc = 'test'

        tr_pred, te_pred = ins.main()
        tr_pred.columns = [ins.regressortype]
        te_pred.columns = [ins.regressortype]

        self.tr = pd.concat([self.tr, tr_pred], axis=1)
        self.te = pd.concat([self.te, te_pred], axis=1)

if __name__ == '__main__':
    # verb = False
    desc = input('\ndescription: ')

    sta = stacking(desc)
    sta.getdatasets()
    sta.extenddata(LinearR)
    sta.extenddata(MLPR)
    sta.extenddata(GBDT)

    gbdt = GBDT(desc)
    gbdt.verb = True
    gbdt.main(sta.tr, sta.te)
