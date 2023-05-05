import numpy as np
import pandas as pd
import pickle
import os
import datetime

class baseClass():
    rand = 1
    verb = False
    # sourcetype = 'nan'

    fdir = os.path.dirname(__file__)
    source_dir = os.path.join(fdir, '../sources/processed')
    export_path = os.path.join(fdir, '../export')

    def __init__(self, desc):
        self.desc = desc

    def main(self, train =None, test =None):
        '''
        trainとtestを与えると全てを終わらせる
        '''
        print('\n===========================================================\n')

        if train is None:
            train = pickle.load(open(os.path.join(self.source_dir, self.sourcetype+'_train.pkl'), 'rb'))
            test = pickle.load(open(os.path.join(self.source_dir, self.sourcetype+'_test.pkl'), 'rb'))

        # データのロードと学習
        x = train.drop(columns=['cover'])
        y = train['cover']

        model, result_df, tr_pred = self.training_cv(x, y)
        sub_df = pd.DataFrame({'index': [i for i in range(len(test))], 'pred': self.getprediction(test, model)['pred']})

        # ファイルの出力
        if self.desc != 'test':
            date = str(datetime.datetime.now().strftime('%m_%d_%H:%M'))
            backup_path = os.path.join(self.export_path, date + '_' + self.regressortype)
            os.mkdir(backup_path)

            pd.concat([train, self.getprediction(train, model)], axis=1).to_csv(os.path.join(backup_path, 'prediction.csv'))
            result_df.to_csv(os.path.join(backup_path, 'result.csv'))
            sub_df.to_csv(os.path.join(backup_path, 'submit.csv'), header=False, index=False)
            pickle.dump(model, open(os.path.join(backup_path, self.regressortype + '_model.pkl'), 'wb'))

            f = open(os.path.join(backup_path, 'description.txt'), 'w')
            f.write(self.desc)
            f.close()
            print('files exported.')
            self.backup_path = backup_path

        if self.verb == True:
            print('\n===========================================================\n')
            gr = result_df.loc[:, ['error', 'type']].groupby('type')
            print(gr.describe())
            print()
        else:
            print(self.regressortype +   ':')
        
        print('validation score: ' + str(np.sqrt(result_df.loc[result_df['type']=='valid', 'error'].mean())))
        print('overfitting: ' + str(np.sqrt(result_df.loc[result_df['type']=='valid', 'error'].mean())/np.sqrt(result_df.loc[result_df['type']=='train', 'error'].mean())))

        self.model = model
        self.col = x.drop(columns='valfold').columns

        sub_df.set_index('index', inplace=True)

        return tr_pred, sub_df

    def training_cv(self, x, y):
        '''
        クロスバリデーションで予測を行う関数
        xとyを与えるとmodelが入ったリストを返す
        '''
        model = []
        result_df = pd.DataFrame()
        prediction = pd.DataFrame()

        for i in range(4):
            tr_x = x.loc[x['valfold']!=i, :].drop(columns='valfold')
            va_x = x.loc[x['valfold']==i, :].drop(columns='valfold')
            tr_y = y.loc[x['valfold']!=i]
            va_y = y.loc[x['valfold']==i]

            mod = self.training(tr_x, tr_y, va_x, va_y)
            model.append(mod)

            result_df = pd.concat([result_df, pd.DataFrame({'index': tr_y.index, 'cover': tr_y, 'pred': self.predict(mod, tr_x), 'type': ['train']*len(tr_y)})])
            result_df = pd.concat([result_df, pd.DataFrame({'index': va_y.index, 'cover': va_y, 'pred': self.predict(mod, va_x), 'type': ['valid']*len(va_y)})])
            
            prediction = pd.concat([prediction, pd.DataFrame({'index': va_x.index, 'pred': self.predict(mod, va_x)})])
        
        prediction.set_index('index', inplace=True)

        result_df['error'] = (result_df['cover'] - result_df['pred'])**2
        result_df.sort_values('type', inplace=True)

        return model, result_df, prediction

    # クロスバリデーションでの予測を得る関数
    def getprediction(self, x, model):
        '''
        予測を得る関数
        特徴量とモデルを渡すと各モデルの予測値とその平均の入ったdataframeを返す
        '''
        result = []
        for i in range(len(model)):
            result.append(list(self.predict(model[i], x)))
        
        result = pd.DataFrame(result, index=['pred_'+str(i) for i in range(4)]).T
        result['pred'] = result.mean(axis=1)

        return result
