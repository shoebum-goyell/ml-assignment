import pandas as pd
import numpy
class Fischers:
    def __init__(self, ds):
        self.ds = ds

    def task1(self,seed):
        training= self.ds.sample(frac=0.67, random_state=seed)
        test = self.ds.drop(training.index)
        test_class = test['diagnosis']
        test = test.drop(['id', 'diagnosis'], axis=1)
        test = test.to_numpy()
        feature_ds1= training[training['diagnosis'] == 'M'].loc[:, ~training.columns.isin(['id', 'diagnosis'])].to_numpy()
        feature_ds2= training[training['diagnosis'] == 'B'].loc[:, ~training.columns.isin(['id', 'diagnosis'])].to_numpy()
        columns = training.columns
        c_types = training.dtypes
        mean1 = []
        mean2 = []
        for i in range(len(columns)):
            if(i!=0):
                if(c_types[i] == float):
                    mean1.append(training.loc[training['diagnosis'] == 'M', columns[i]].mean())
                    mean2.append(training.loc[training['diagnosis'] == 'B', columns[i]].mean())
        mean1 = numpy.array([numpy.array(mean1)])
        mean2 = numpy.array([numpy.array(mean2)])
        

        sum1 = numpy.zeros((len(columns)-2, len(columns)-2))
        for i in feature_ds1:
            i = [numpy.array(i)]
            diff = i - mean1
            sum1 = sum1 + numpy.matmul(numpy.transpose(diff), diff)
        a = sum1/(len(feature_ds1))

        sum2 = numpy.zeros((len(columns)-2, len(columns)-2))
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
        for i in range(len(test)):
            testproj = numpy.dot(test[i], numpy.transpose(w))
            prob1 = (1/(proj1.std()*numpy.sqrt(2*numpy.pi))) * numpy.exp((-0.5) * numpy.square((testproj[0]-proj1.mean())/proj1.std()))
            prob2 = (1/(proj2.std()*numpy.sqrt(2*numpy.pi))) * numpy.exp((-0.5) * numpy.square((testproj[0]-proj2.mean())/proj2.std()))
            prediction = ''
            if(prob1 >= prob2):
                prediction = 'M'
            else:
                prediction = 'B'

            if(prediction == test_class.to_numpy()[i]):
                accuracy = accuracy+1
        accuracy = accuracy/len(test)
        print("accuracy = ",accuracy*100,"%")
    


       

            




        
                

        