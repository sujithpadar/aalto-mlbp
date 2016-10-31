import pandas as pd

def ConfusionMatrix(actual,predicted):
    df = pd.DataFrame(data={'actual' : actual,'predicted' : predicted})
    countmatrix = df.groupby(['actual', 'predicted']).size()

    accuracy = round((countmatrix[0][0]+countmatrix[1][1])/df.shape[0],4)
    postive_pred_accuracy = round(countmatrix[1][1]/(countmatrix[1][1]+countmatrix[0][1]),4)
    negative_pred_accuracy = round(countmatrix[0][0]/(countmatrix[0][0]+countmatrix[1][0]),4)
    postive_pred_rate = round(countmatrix[1][1]/(countmatrix[1][0]+countmatrix[1][1]),4)
    negative_pred_rate = round(countmatrix[0][0]/(countmatrix[0][0]+countmatrix[0][1]),4)

    summary = {'0CountMatrix' : countmatrix,
               'Accuracy' : accuracy,
               'Positive Pred Accuracy': postive_pred_accuracy,
               'Positive Pred Rate': postive_pred_rate,
               'Negative Pred Accuracy': negative_pred_accuracy,
               'Negative Pred Rate': negative_pred_rate}
    return summary