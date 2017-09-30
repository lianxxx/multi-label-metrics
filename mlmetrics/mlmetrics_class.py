#sample input matrix format - true labels and predicted labels
#y_true = np.array([[1,0,1,1],[0,1,0,0],[0,0,0,0],[0,1,1,0]])
#y_pred = np.array([[0,0,1,1],[1,1,0,1],[0,1,0,0],[1,1,1,1]]

#function to calculate some comman values for all the metrics
class mlmetrics:
    def _init_(self, y_true, y_pred):
        if target.shape != prediction.shape :
            raise ValueError('Wrongs predictions metrics!')

        real_pos = real_neg = pred_pos = pred_neg  = true_pos = true_neg = []

        for i in range(y_true.shape[0]):
            # real values - RP and RN
            self.real_pos = np.asarray(np.append(self.real_pos,np.logical_and(y_true[i], y_true[i]).sum()), dtype=np.int64).reshape(-1,1)
            self.real_neg = np.asarray(np.append(self.real_neg,np.logical_and(np.logical_not(y_true[i]),np.logical_not(y_true[i])).sum()), dtype=np.int64).reshape(-1,1)

            # predicted values - PP and PN
            self.pred_pos = np.asarray(np.append(self.pred_pos,np.logical_and(y_pred[i], y_pred[i]).sum()),dtype=np.int64).reshape(-1,1)
            self.pred_neg = np.asarray(np.append(self.pred_neg,np.logical_and(np.logical_not(y_pred[i]), np.logical_not(y_pred[i])).sum()),dtype=np.int64).reshape(-1,1)

            # true labels - TP and TN
            self.true_pos = np.asarray(np.append(self.true_pos,np.logical_and(y_true[i], y_pred[i]).sum()),dtype=np.int64).reshape(-1,1)
            self.true_neg = np.asarray(np.append(self.true_neg,np.logical_and(np.logical_not(y_true[i]), np.logical_not(y_pred[i])).sum()),dtype=np.int64).reshape(-1,1)



    #function to resolve divide by zero error and accept the value 0 when divided by 0
    def divideZero(self, value_a, value_b):
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.true_divide( value_a, value_b )
            result[ ~ np.isfinite( result )] = 0
        return result

    def accuracy(self):
        #return the accuracy - example based
        score = (self.true_pos + self.true_neg)/(self.pred_pos + self.pred_neg)
        score = np.mean(score)
        return score


    def precision(self):
        #return precision - example based
        score = divideZero(self.true_pos, self.pred_pos)
        score = np.mean(score)
        return score

    def recall(self):
        #return precision - example based
        score = divideZero(self.true_pos, self.real_pos)
        score = np.mean(score)
        return score


    def fscore(self,beta = 1):
    	#return f(beta)score - example based : default beta value is 1
        prec, rec = precision(), recall()
        beta_val = beta*beta
        score = ((1+beta_val)*(prec*rec))/(beta_val*(prec+rec))
        return score


    def hamloss(self):
    	#return hamming loss - example based
        hamloss = list()
        for i in range(y_true.shape[0]):
            hamloss = np.asarray(np.append(hamloss,np.logical_xor(y_true[i], y_pred[i]).sum()), dtype=np.int64).reshape(-1,1)
        score = (hamloss.sum())/((y_true.shape[0])*(y_true.shape[1]))
        return score


    def subset(self):
    	#return subset accuracy - example based
        subset_matrix = list()
        for i in range(y_true.shape[0]):
            subset_matrix = np.asarray(np.append(subset_matrix, np.array_equal(y_true[i],y_pred[i])), dtype=np.int64).reshape(-1,1)
        score = (subset_matrix.sum())/((y_true.shape[0])*(y_true.shape[1]))
        return score

    def zeroloss(self):
        #return new array with removed element having all zero in y_true
        condition = list()
        index = list()
        for i in range(y_true.shape[0]):
            new_true = new_pred = list()
            condition = np.logical_and(y_true[i],y_true[i]).sum()
            if (condition==0):
                index = np.asarray(np.append(index,i), dtype = np.int64)

            new_true = np.delete(y_true,index, axis = 0)
            new_pred = np.delete(y_pred,index, axis = 0)
        return new_true, new_pred

    def microprecision(self):
        #return micro-precision
        score = self.true_pos.sum()/self.pred_pos.sum()
        return score

    def microrecall(self):
        #return micro-recall
        score = self.true_pos.sum()/self.real_pos.sum()
        return score

    def microfscore(self,beta = 1):
        #return micro-fscore
        prec, rec = microprecision(), microrecall()
        beta_val = beta*beta
        score = ((1+beta_val)*(prec*rec))/(beta_val*(prec+rec))
        return score

    def macroprecision(self):
        #return macro-precision
        score = divideZero(self.true_pos, self.pred_pos)
        return score

    def macrorecall(self):
        #return macro-recall
        score = divideZero(self.true_pos, self.real_pos)
        return score

    def macrofscore(self,beta = 1):
        #return macro-fscore
        prec, rec = macroprecision(), macrorecall()
        beta_val = beta*beta
        score = divideZero(((1+beta_val)*(prec*rec)),(beta_val*(prec+rec)))
        score = np.mean(score)
        return score
