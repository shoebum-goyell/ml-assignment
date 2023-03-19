import pandas as pd
import numpy
class Perceptron:
    def __init__(self, ds):
        self.ds = ds

    def task1(self, seed, epochs):
        training= self.ds.sample(frac=0.67, random_state=seed)
        test = self.ds.drop(training.index)
        test_class = test['diagnosis'].to_numpy()
        test = test.drop(['id', 'diagnosis'], axis=1)
        test = test.to_numpy()
        training_class = training['diagnosis'].to_numpy()
        feature_ds= training.drop(['id', 'diagnosis'], axis=1).to_numpy()
        columns = training.columns
        w = numpy.zeros(len(columns)-2, dtype=int)
        for j in range(epochs):
            m = 0
            for i in range(len(feature_ds)):
                var = -1
                if(training_class[i] == 'M'):
                    var = 1
                if(numpy.dot(w, feature_ds[i])*var <=0):
                    w = w + (feature_ds[i]*var)
                    m = m + 1
            # if(m/len(feature_ds) < 0.001):
            #     print((len(feature_ds) - m) / len(feature_ds))
            #     break
            accuracy = 0
            for i in range(len(test)):
                var = -1
                if(test_class[i] == 'M'):
                    var = 1
                if(numpy.dot(w, test[i])*var > 0):
                    accuracy = accuracy + 1
            print("Accuracy: ", accuracy/len(test))

        
        
    


       

            

