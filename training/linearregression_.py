# coding: utf-8
from sklearn.linear_model import LinearRegression
import basecodes

class LinearR(basecodes.baseClass):
    regressortype = 'LinearRegressor'
    sourcetype = 'filled'
    rand = basecodes.baseClass.rand
    verb = False

    def __init__(self, desc):
        super().__init__(desc)
        
    def training(self, tr_x, tr_y, va_x, va_y):   
        model = LinearRegression()
        model.fit(tr_x.values, tr_y.values)

        return model

    def predict(self, model, x):
        return model.predict(x.values)

if __name__ == '__main__':
    desc = input('\ndescription: ')
    lr = LinearR(desc)
    # lr.verb = True
    _, _ = lr.main()
