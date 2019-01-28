import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .metrics import confusion_matrix


def plot_confusion_heatmap(df, xlabel='Predicted labels', ylabel='True labels', xticks_rotation=45, yticks_rotation=0,
                           fontsize=14):
    matrix, class_names, _ = confusion_matrix(df)

    df_cm = pd.DataFrame(
        matrix, index=class_names, columns=class_names,
    )
    heatmap = sns.heatmap(df_cm, annot=True, fmt='g')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=xticks_rotation, ha='right',
                                 fontsize=fontsize)
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=yticks_rotation, ha='right',
                                 fontsize=fontsize)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
