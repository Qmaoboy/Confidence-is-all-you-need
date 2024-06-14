
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from random import randint
def Cal_roc_curve(y_true,y_scores,data_name):

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Compute AUROC
    auc = roc_auc_score(y_true, y_scores)
    print(f"AUROC: {auc:.4f}")

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(f"PACE/picture/{data_name}_roc_curve.png")
    # plt.show()
    plt.clf()


def prac1():
    datapath="PACE/response_result/Evaluate_Result_20240601_No_shuffle.json"
    with open(datapath,'r') as f:
        data=json.load(f)
    acc_list=[]
    paceconf_list=[]
    conf_list=[]
    stretagy="vanilla"
    for i in data:
        if i['dataset']=='din0s/asqa' and i['Stratagy']== stretagy and i['api_model']=="gpt-3.5-turbo-0125" and i['sim_model']== 'Cos_sim' and i['acc_model']=='f1':
            acc_list.append(i['Accuracy'])
            paceconf_list.append(i['Pace_Conf'])
            conf_list.append(i['Conf'])

    acc=np.array(acc_list[0])
    print(acc)
    conf=np.array(conf_list[0])
    pace_conf=np.array(paceconf_list[0])
    accuracy=np.mean(acc)
    print(accuracy)
    # y_true=np.where(acc < 0.7,0,1) ## change to binary
    # print(y_true)
    # print(len(acc))
    # print(np.sum(y_true))

    # Cal_roc_curve(y_true,pace_conf,f"{stretagy}_pace_conf")
    # Cal_roc_curve(y_true,conf,f"{stretagy}_conf")



if __name__=="__main__":
    prac1()
    # print(randint(0,100))
