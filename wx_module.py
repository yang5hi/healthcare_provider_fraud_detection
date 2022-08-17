# coding: utf-8

class Metric_pipeline(object):
    def __init__(self, X_train, y_train, X_test,y_test,model):
            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            self.y_test = y_test
            self.model = model
   
        
    def roc_auc_score(self):
        
        import matplotlib.pyplot as plt
        
        from sklearn.metrics import roc_auc_score
        roc_auc_score_train = roc_auc_score(self.y_train, self.model.predict_proba(self.X_train)[:, 1])  
        roc_auc_score_test  = roc_auc_score(self.y_test, self.model.predict_proba(self.X_test)[:, 1])
        print ('roc_auc_score for the train dataset: {:.3f}'.format(roc_auc_score_train))
        from sklearn.metrics import plot_roc_curve 
        
        #plt.figure(1).clf()
        plot_roc_curve(self.model, self.X_train, self.y_train)
        plt.title('Train ROC Curve')
        
        plt.show()
        print ('roc_auc_score for the test dataset: {:.3f}'.format(roc_auc_score_test))
        plot_roc_curve(self.model, self.X_test, self.y_test)
        plt.title('Test ROC Curve')
        plt.show()
        
    def PR_AUC(self):
        
        import matplotlib.pyplot as plt
        ####train plot
        y_train_proba = self.model.predict_proba(self.X_train)
        y_train_score = y_train_proba[:, 1]
        from sklearn.metrics import precision_recall_curve
        from sklearn.metrics import auc,plot_precision_recall_curve
         
        precision, recall, thresholds = precision_recall_curve(self.y_train, y_train_score)
      
        auc_precision_recall = auc(recall, precision)
        print('train'+' PR-AUC is {:.3f}'.format(auc_precision_recall))
        
        plt.plot(recall, precision)
        plt.xlabel('Recall(Positive label:1)')
        plt.ylabel('Precision(Positive label:1)')
        plt.title('Precision-Recall Curve'+' of '+'train')
        plt.show()
        
        
        ####test plot
        y_test_proba = self.model.predict_proba(self.X_test)
        y_test_score = y_test_proba[:, 1]
        precision, recall, thresholds = precision_recall_curve(self.y_test, y_test_score)
        auc_precision_recall = auc(recall, precision)
        print('test'+' PR-AUC is {:.3f}'.format(auc_precision_recall))
   
        plt.plot(recall, precision)
        plt.xlabel('Recall(Positive label:1)')
        plt.ylabel('Precision(Positive label:1)')
        plt.title('Precision-Recall Curve'+' of '+'test')
        plt.show()
        
        
    def classification_report(self):
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        from sklearn.metrics import classification_report
        print('---------------------Train Classification Report--------------------------------')
        print(classification_report(self.y_train, y_train_pred))
        print('---------------------Test Classification Report--------------------------------')
        print(classification_report(self.y_test, y_test_pred))
    
    def metrics(self):
        from sklearn.metrics import precision_score
        from sklearn.metrics import average_precision_score
        from sklearn.metrics import roc_auc_score
        import pandas as pd
        
        res=[]
        
        #####roc_auc_score
        roc_auc_score_train = roc_auc_score(self.y_train, self.model.predict_proba(self.X_train)[:, 1])  
        roc_auc_score_test  = roc_auc_score(self.y_test, self.model.predict_proba(self.X_test)[:, 1])
        res.append(['roc_auc_score(train)','{:.3f}'.format(roc_auc_score_train)])
        res.append(['roc_auc_score(test)','{:.3f}'.format(roc_auc_score_test)])
        
        
        
        
        #####PR
        y_train_proba = self.model.predict_proba(self.X_train)
        y_train_score= y_train_proba[:, 1]
        from sklearn.metrics import precision_recall_curve
        from sklearn.metrics import auc,plot_precision_recall_curve

        precision, recall, thresholds = precision_recall_curve(self.y_train, y_train_score)
        auc_precision_recall = auc(recall, precision)
        res.append(['PR-AUC(train)','{:.3f}'.format(auc_precision_recall)])
        
        
        

        y_test_proba = self.model.predict_proba(self.X_test)
        y_test_score = y_test_proba[:, 1]
        precision, recall, thresholds = precision_recall_curve(self.y_test, y_test_score)
        auc_precision_recall = auc(recall, precision)
        res.append(['PR-AUC(test)','{:.3f}'.format(auc_precision_recall)])
        
        #precision recall
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        res.append(['average_precision(train)','{:.3f}'.format(average_precision_score(self.y_train,y_train_score))])
        res.append(['average_precision(test)','{:.3f}'.format(average_precision_score(self.y_test,y_test_score))])
        res.append(['precision_score(train)','{:.3f}'.format(precision_score(self.y_train, y_train_pred, average='weighted'))])
        res.append(['precision_score(test)','{:.3f}'.format(precision_score(self.y_test, y_test_pred, average='weighted'))])
        return pd.DataFrame(res)