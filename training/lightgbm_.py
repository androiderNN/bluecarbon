# coding: utf-8
import lightgbm as lgb
import pandas as pd
import os

import basecodes

class GBDT(basecodes.baseClass):
    regressortype = 'Lightgbm'
    sourcetype = 'nan'
    # rand = basecodes.baseClass.rand
    rand = 435
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbose': -1,
        'random_state': rand,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'feature_fraction': 0.8,
        'min_data_in_leaf': 10,
        'lambda_l1': 0.2,
        'lambda_l2': 0.2
        }

    def __init__(self, desc):
        super().__init__(desc)

    def training(self, tr_x, tr_y, va_x, va_y):        
        lgb_train = lgb.Dataset(tr_x, tr_y)
        lgb_val = lgb.Dataset(va_x, va_y)

        model = (lgb.train(
            self.params,
            lgb_train,
            valid_sets=[lgb_train, lgb_val],
            valid_names=['train', 'val'],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(stopping_rounds=1, verbose=self.verb)]
        ))

        return model

    def predict(self, model, x):
        return model.predict(x, predict_disable_shape_check=True)
    
    def main(self, train =None, test =None):
        tr_pred, sub_df = super().main(train, test)
        
        imp = [self.model[i].feature_importance(importance_type='gain') for i in range(4)]
        imp = [sum(l) for l in zip(*imp)]
        imp_df = pd.DataFrame({'col': self.col, 'importance': imp})
        imp_df.sort_values('importance', ascending=False, inplace=True)
    
        if self.verb == True:
            print()
            print(imp_df)
            print()

        if self.desc != 'test':
            imp_df.to_csv(os.path.join(self.backup_path, 'importance.csv'))

        return tr_pred, sub_df

if __name__ == '__main__':
    desc = input('\ndescription: ')
    gbdt = GBDT(desc)
    gbdt.verb = True
    gbdt.main()