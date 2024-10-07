import seaborn as sns
from load_save import *
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
from Confusion_mat import *

def bar_plot(label, value, metric):

    fig = plt.figure(figsize=(7, 6))
    colors = ['maroon', 'green', 'blue', 'orange', 'purple']

    # creating the bar plot
    plt.bar(label, value, color=colors,
            width=0.4)

    plt.xlabel("Method")
    plt.ylabel(metric)
    plt.savefig('./Results/'+metric+'.png', dpi=100)
    plt.show(block=False)

def plot_res():

    table1=load('Table1')

    val1 = np.array(table1)
    mthod=['Logistic_regression','SVM','DT','xgboost','PROPOSED']
    metrices_plot=['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F-Measure', 'MCC', 'NPV', 'FPR', 'FNR']

    # Bar plot
    for i in range(len(metrices_plot)):
        bar_plot(mthod, val1[i, :], metrices_plot[i])

    print('Testing Metrices 1')
    tab = pd.DataFrame(val1, index=metrices_plot, columns=mthod)
    excel_file_path = './Results/Classi_RESULT.xlsx'
    tab.to_excel(excel_file_path, index=metrices_plot)  # Specify index=False to exclude index column
    print(tab)


def bar_plot_opti(label, value, metric):
    fig = plt.figure(figsize=(7, 6))
    colors = ['maroon', 'green', 'blue', 'orange', 'purple']

    # creating the bar plot
    plt.bar(label, value, color=colors,
            width=0.4)

    plt.xlabel("Method")
    plt.ylabel(metric)
    plt.savefig('./Results/'+metric+'optimizer.png', dpi=100)
    plt.show(block=False)



def plot_res2():

    table2=load('Table2')

    val2= np.array(table2)
    mthod1=['Logistic_regression','SVM','DT','xgboost','PROPOSED']
    metrices_plot=['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F-Measure', 'MCC', 'NPV', 'FPR', 'FNR']

    # Bar plot
    for i in range(len(metrices_plot)):
        bar_plot_opti(mthod1, val2[i, :], metrices_plot[i])


    print('Testing Metrices 2')
    tab = pd.DataFrame(val2, index=metrices_plot, columns=mthod1)
    excel_file_path = './Results/OPTI_RESULT.xlsx'
    tab.to_excel(excel_file_path, index=metrices_plot)  # Specify index=False to exclude index column

    print(tab)

    '''
    DOS = 0     "back.": 0, "land.": 0, "neptune.": 0, "pod.": 0, "smurf.": 0, "teardrop.": 0,

    U2R = 1       "buffer_overflow.": 1, "perl.": 1, "loadmodule.": 1, "rootkit.": 1

    R2L = 2    "ftp_write.": 2, "guess_passwd.": 2, "imap.": 2, "multihop.": 2,
                      "phf.": 2, "spy.": 2, "warezclient.": 2, "warezmaster.": 2,
                      ''snmpguess',


    Probing = 3       "ipsweep.": 3, "nmap.": 3, "portsweep.": 3, "satan.": 3,'mscan':3

    normal = 4      normal = 4
    '''

    y_test = load("y_test")
    ypred = load('ypred')

    # Compute the first confusion matrix (cm) and plot it
    cm = confusion_matrix(y_test, ypred)
    plt.figure(figsize=(6, 6))

    sns.heatmap(cm,
                annot=True,
                fmt='g',
                xticklabels=['DOS ', 'U2R', 'R2L', 'Probing', 'normal'],
                yticklabels=['DOS ', 'U2R', 'R2L', 'Probing', 'normal'],
                cmap='summer')
    plt.ylabel('Prediction', fontsize=13)
    plt.xlabel('Actual', fontsize=13)
    plt.title('Dataset1', fontsize=13)
    plt.savefig('./Results/Confusion matrix.png')
    plt.tight_layout()  # Ensure subplots don't overlap
    plt.show()

# plot_res2()
#
# plot_res()