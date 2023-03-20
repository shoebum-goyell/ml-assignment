import pandas as pd
import numpy
class Fischers:
    def __init__(self, training, test):
        self.training = training
        self.test = test

    def fischers(self):
        test_class = self.test['diagnosis']
        self.test = self.test.drop(['id', 'diagnosis'], axis=1)
        self.test = self.test.to_numpy()
        feature_ds1= self.training[self.training['diagnosis'] == 'M'].loc[:, ~self.training.columns.isin(['id', 'diagnosis'])]
        columns = feature_ds1.columns
        feature_ds1 = feature_ds1.to_numpy()
        feature_ds2= self.training[self.training['diagnosis'] == 'B'].loc[:, ~self.training.columns.isin(['id', 'diagnosis'])].to_numpy()
        mean1 = []
        mean2 = []
        for i in range(len(columns)):
            mean1.append(self.training.loc[self.training['diagnosis'] == 'M', columns[i]].mean())
            mean2.append(self.training.loc[self.training['diagnosis'] == 'B', columns[i]].mean())
        mean1 = numpy.array([numpy.array(mean1)])
        mean2 = numpy.array([numpy.array(mean2)])
        

        sum1 = numpy.zeros((len(columns), len(columns)))
        for i in feature_ds1:
            i = [numpy.array(i)]
            diff = i - mean1
            sum1 = sum1 + numpy.matmul(numpy.transpose(diff), diff)
        a = sum1/(len(feature_ds1))

        sum2 = numpy.zeros((len(columns), len(columns)))
        for i in feature_ds2:
            i = [numpy.array(i)]
            diff = i - mean2
            sum2 = sum2 + numpy.matmul(numpy.transpose(diff), diff)
        b = sum2/(len(feature_ds2))

        sw = a + b
        w = numpy.matmul(mean2 - mean1, numpy.linalg.inv(sw))
        proj1 = numpy.dot(feature_ds1, numpy.transpose(w))
        proj2 = numpy.dot(feature_ds2, numpy.transpose(w))

        accuracy = 0
        for i in range(len(self.test)):
            testproj = numpy.dot(self.test[i], numpy.transpose(w))
            prob1 = (1/(proj1.std()*numpy.sqrt(2*numpy.pi))) * numpy.exp((-0.5) * numpy.square((testproj[0]-proj1.mean())/proj1.std()))
            prob2 = (1/(proj2.std()*numpy.sqrt(2*numpy.pi))) * numpy.exp((-0.5) * numpy.square((testproj[0]-proj2.mean())/proj2.std()))
            prediction = ''
            if(prob1 >= prob2):
                prediction = 'M'
            else:
                prediction = 'B'

            if(prediction == test_class.to_numpy()[i]):
                accuracy = accuracy+1
        accuracy = accuracy/len(self.test)
        # print("accuracy = ",accuracy*100,"%")
        return accuracy*100
    


       

            




        
                

        