import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import lightgbm as lgb
import pickle
import os
import datetime

rand = 234
source_dir = '../../sources/processed'
export_path = '../../export'

desc = input('description: ')
print()

train = pickle.load(open(os.path.join(source_dir, 'lgb_train.pkl'), 'rb'))
test = pickle.load(open(os.path.join(source_dir, 'lgb_test.pkl'), 'rb'))

x = train.drop(columns=['cover'])
y = train['cover']

# tr_x, va_x, tr_y, va_y = train_test_split(train.drop(columns=['cover']), train['cover'], test_size=0.3, random_state=rand)
kf = KFold(n_splits=4, shuffle=True, random_state=rand)
model = []
for tr_idx, va_idx in kf.split(train):
    tr_x, va_x = x.iloc[tr_idx], x.iloc[va_idx]
    tr_y, va_y = y.iloc[tr_idx], y.iloc[va_idx]

    lgb_train = lgb.Dataset(tr_x, tr_y)
    lgb_val = lgb.Dataset(va_x, va_y)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbose': -1,
        'random_state': rand
    }

    model.append(lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train', 'val'],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=3, verbose=True)]
    ))

    print()

def getprediction(x, model):
    result = []
    for i in range(4):
        result.append(list(model[i].predict(x, predict_disable_shape_check=True)))
    
    result = pd.DataFrame(result, index=['pred_'+str(i) for i in range(4)]).T
    result['pred'] = result.mean(axis=1)

    return result

sub_df = pd.DataFrame({'index': [i for i in range(len(test))], 'pred': getprediction(test, model)['pred']})

va_y.index = [i for i in range(len(va_y))]
va_y.name = 'cover'
tmp = getprediction(va_x, model)
tmp.index = [i for i in range(len(va_y))]

result_df = pd.concat([va_y, tmp], axis=1)
# result_df.columns = ['cover'] + list(tmp.columns)
result_df['error'] = (result_df['cover'] - result_df['pred'])**2

imp = model[3].feature_importance(importance_type='gain')
imp_df = pd.DataFrame({'col': tr_x.columns, 'importance': imp})
imp_df.sort_values('importance', ascending=False, inplace=True)
print(imp_df)
print()

if desc != 'test':
    date = str(datetime.datetime.now().strftime('%m_%d_%H:%M'))
    backup_path = os.path.join(export_path, date)
    os.mkdir(backup_path)

    imp_df.to_csv(os.path.join(backup_path, 'importance.csv'))
    va_x.to_csv(os.path.join(backup_path, 'result.csv'))
    sub_df.to_csv(os.path.join(backup_path, 'submit.csv'), header=False, index=False)
    pickle.dump(model, open(os.path.join(backup_path, 'lgb_model.pkl'), 'wb'))

    f = open(os.path.join(backup_path, 'description.txt'), 'w')
    f.write(desc)
    f.close()

print('validation score: ' + str(np.sqrt(result_df['error'].mean())))
