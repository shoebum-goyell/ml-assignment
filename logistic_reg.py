import numpy
import matplotlib.pyplot as plt

def sigmoid(x):
    arr = []
    for i in x:
       arr.append(1/(1+numpy.exp(-i)))
    return arr

class LogisticRegression():
    def __init__ (self, lr, iters, threshold):
        self.lr = lr
        self.iters = iters
        self.threshold = threshold
        self.weights = None
        self.bias = None

    def fit(self, x, y, descent, isgraphshown):
        n_samples, n_features = numpy.shape(x)
        self.weights = numpy.zeros(n_features)
        self.bias = 0
        if(descent == 'bgd'):
            cost_arr = []
            for i in range(self.iters):
                cost = 0
                linear_predictions = numpy.dot(x, self.weights) + self.bias
                predictions = sigmoid(numpy.array(linear_predictions).astype(float))

                for j in range(len(predictions)):
                    if(y[j] == 1):
                        cost = cost + (-1)*(numpy.log(predictions[j]+1e-9))
                    else:
                        cost = cost + (-1)*(numpy.log(1-predictions[j]+1e-9))
                cost = cost/len(predictions)
                cost_arr.append(cost)

                dw = (1/n_samples) * numpy.dot(numpy.transpose(x), (predictions - y))
                db = (1/n_samples) * numpy.sum(predictions - y)
                self.weights = self.weights - (self.lr*dw)
                self.bias = self.bias - (self.lr*db)

            cost_arr = numpy.array(cost_arr)
            plt.plot(cost_arr)
            plt.xlabel("iterations")
            plt.ylabel("cost")
            if(isgraphshown):
                plt.show()

        elif(descent == 'sgd'):
            cost_arr = []
            for i in range(self.iters):
                cost = 0
                for j in range(len(x)):
                    linear_predictions = numpy.dot(x[j], self.weights) + self.bias
                    predictions = sigmoid([linear_predictions])
                    if(y[j] == 1):
                        cost = cost + (-1)*(numpy.log(predictions[0]+1e-9))
                    else:
                        cost = cost + (-1)*(numpy.log(1-predictions[0]+1e-9))
                    cost = cost/len(predictions)
                    cost_arr.append(cost)
                    dw = numpy.dot(numpy.transpose(x[j]), (predictions[0] - y[j]))
                    db = numpy.sum(predictions[0] - y[j])
                    self.weights = self.weights - (self.lr*dw)
                    self.bias = self.bias - (self.lr*db)
            
            cost_arr = numpy.array(cost_arr)
            cost_arr_new = []
            for i in range(len(cost_arr)):
                if(i%200 == 0):
                    cost_arr_new.append(cost_arr[i])
            plt.plot(cost_arr_new)
            plt.xlabel("iterations / 200")
            plt.ylabel("cost")
            if(isgraphshown):
                plt.show()

        elif(descent == 'mbgd'):
            cost_arr = []
            mini = 100
            for i in range(self.iters):
                cost = 0
                j = mini
                while(j <= len(x)):
                    ymini = y[j-mini:j]
                    linear_predictions = numpy.dot(x[j-mini:j], self.weights) + self.bias
                    predictions = sigmoid(numpy.array(linear_predictions).astype(float))

                    for k in range(len(predictions)):
                        if(ymini[k] == 1):
                            cost = cost + (-1)*(numpy.log(predictions[k]+1e-9))
                        else:
                            cost = cost + (-1)*(numpy.log(1-predictions[k]+1e-9))
                    cost = cost/len(predictions)
                    cost_arr.append(cost)

                    dw = (1/n_samples) * numpy.dot(numpy.transpose(x[j-mini:j]), (predictions - ymini))
                    db = (1/n_samples) * numpy.sum(predictions - ymini)
                    self.weights = self.weights - (self.lr*dw)
                    self.bias = self.bias - (self.lr*db)
                    if(j + mini < len(x)):
                        j = j+mini
                    elif(j == len(x)):
                        j = len(x)+2
                    else:
                        j = len(x)
            cost_arr = numpy.array(cost_arr)
            cost_arr_new = []
            for i in range(len(cost_arr)):
                if(i%50 == 0):
                    cost_arr_new.append(cost_arr[i])
            plt.plot(cost_arr_new)
            plt.xlabel("iterations / 50")
            plt.ylabel("cost")
            if(isgraphshown):
                plt.show()

        
    def predict(self, x):
        linear_predictions = numpy.dot(x, self.weights) + self.bias
        y_pred = sigmoid(linear_predictions)
        class_predictions = []
        for i in y_pred:
            if(i <= self.threshold):
                class_predictions.append(0)
            else:
                class_predictions.append(1)

        return class_predictions
    

 
