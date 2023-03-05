# coding: utf-8
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
import pickle
import os
import datetime

# cvでの学習用関数
# tr_x, va_x, tr_y, va_y = train_test_split(train.drop(columns=['cover']), train['cover'], test_size=0.3, random_state=rand)
def training_cv(x, y, rand):
    '''
    クロスバリデーションで予測を行う関数
    xとyを与えるとmodelが入ったリストを返す
    '''
    kf = KFold(n_splits=4, shuffle=True, random_state=rand)
    model = []
    result_df = pd.DataFrame()

    for tr_idx, va_idx in kf.split(train):
        tr_x, va_x = x.iloc[tr_idx], x.iloc[va_idx]
        tr_y, va_y = y.iloc[tr_idx], y.iloc[va_idx]

        mod = MLPRegressor(max_iter=10000, hidden_layer_sizes=(50, 50, 10), random_state=rand, early_stopping=True)
        mod.fit(tr_x.values, tr_y.values)
        model.append(mod)

        result_df = pd.concat([result_df, pd.DataFrame({'index': tr_y.index, 'cover': tr_y, 'pred': mod.predict(tr_x.values), 'type': ['train']*len(tr_y)})])
        result_df = pd.concat([result_df, pd.DataFrame({'index': va_y.index, 'cover': va_y, 'pred': mod.predict(va_x.values), 'type': ['valid']*len(va_y)})])

    result_df['pred'].clip(0, 1, inplace=True)
    result_df['error'] = (result_df['cover'] - result_df['pred'])**2
    result_df.sort_values('type', inplace=True)

    return model, result_df

# 予測を得るための関数
def getprediction(x, model):
    '''
    予測を得るための関数
    特徴量とモデルを渡すと各モデルの予測値とその平均の入ったdataframeを返す
    '''
    result = []
    for i in range(len(model)):
        result.append(list(model[i].predict(x.values)))
    
    result = pd.DataFrame(result, index=['pred_'+str(i) for i in range(4)]).T
    result['pred'] = result.mean(axis=1)

    return result

if __name__ == '__main__':
    rand = 2
    fdir = os.path.dirname(__file__)
    source_dir = os.path.join(fdir, '../../sources/processed')
    export_path = os.path.join(fdir, '../../export')

    desc = input('description: ')
    print('\n===========================================================\ntraining...')

    # データのロードと学習
    train = pickle.load(open(os.path.join(source_dir, 'nn_train.pkl'), 'rb'))
    test = pickle.load(open(os.path.join(source_dir, 'nn_test.pkl'), 'rb'))

    x = train.drop(columns=['cover'])
    y = train['cover']

    model, result_df = training_cv(x, y, rand)

    # 予測の取得
    sub_df = pd.DataFrame({'index': [i for i in range(len(test))], 'pred': getprediction(test, model)['pred']})
    sub_df['pred'].clip(0, 1, inplace=True)

    # ファイルの出力
    if desc != 'test':
        date = str(datetime.datetime.now().strftime('%m_%d_%H:%M'))
        backup_path = os.path.join(export_path, date + '_nn')
        os.mkdir(backup_path)

        result_df.to_csv(os.path.join(backup_path, 'result.csv'))
        sub_df.to_csv(os.path.join(backup_path, 'submit.csv'), header=False, index=False)
        pickle.dump(model, open(os.path.join(backup_path, 'nn_model.pkl'), 'wb'))

        f = open(os.path.join(backup_path, 'description.txt'), 'w')
        f.write(desc)
        f.close()
        print('files exported.')

    print('\n===========================================================\n')

    gr = result_df.loc[:, ['error', 'type']].groupby('type')
    print(gr.describe())

    print()
    print('validation score: ' + str(np.sqrt(result_df.loc[result_df['type']=='valid', 'error'].mean())))
    print('overfitting: ' + str(np.sqrt(result_df.loc[result_df['type']=='valid', 'error'].mean())/np.sqrt(result_df.loc[result_df['type']=='train', 'error'].mean())))
    print(result_df.sort_values('cover'))
