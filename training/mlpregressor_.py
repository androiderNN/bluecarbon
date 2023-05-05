# coding: utf-8
from sklearn.neural_network import MLPRegressor
import basecodes

class MLPR(basecodes.baseClass):
    regressortype = 'MLPRegressor'
    sourcetype = 'filled'
    rand = basecodes.baseClass.rand
    verb = False

    def __init__(self, desc):
        super().__init__(desc)
        
    def training(self, tr_x, tr_y, va_x, va_y):   
        model = MLPRegressor(
            max_iter=10000,
            hidden_layer_sizes=(70, 70, 10),
            random_state=self.rand,
            early_stopping=True,
            validation_fraction=0.2,
            verbose=False
            )
        model.fit(tr_x.values, tr_y.values)

        return model

    def predict(self, model, x):
        return model.predict(x.values)

if __name__ == '__main__':
    desc = input('\ndescription: ')
    mr = MLPR(desc)
    mr.verb = True
    _, _ = mr.main()
