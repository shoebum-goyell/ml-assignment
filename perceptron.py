import pandas as pd
import numpy
class Perceptron:
    def __init__(self, training, test):
        self.training = training
        self.test = test

    def perceptron(self, epochs):
        test_class = self.test['diagnosis'].to_numpy()
        self.test = self.test.drop(['id', 'diagnosis'], axis=1)
        self.test = self.test.to_numpy()
        training_class = self.training['diagnosis'].to_numpy()
        feature_ds= self.training.drop(['id', 'diagnosis'], axis=1).to_numpy()
        columns = self.training.columns
        w = numpy.zeros(len(columns)-2, dtype=int)
        accuracy = 0
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
            for i in range(len(self.test)):
                var = -1
                if(test_class[i] == 'M'):
                    var = 1
                if(numpy.dot(w, self.test[i])*var > 0):
                    accuracy = accuracy + 1
        #     print("Accuracy: ", accuracy/len(self.test))
        # print("accuracy = ", accuracy*100/len(self.test), "%")
        return accuracy*100/len(self.test)

        
        
    


       

            

