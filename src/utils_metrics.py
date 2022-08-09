from sklearn import metrics
import seaborn as sns


def plot_confusion_matrix(true_labels, predicted_labels, result_dir):
    confusion_matrix = metrics.confusion_matrix(true_labels, predicted_labels)
    
    fig, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(confusion_matrix, vmin=0, vmax=1, cmap="Blues", square=True, ax=ax)
    fig.savefig(os.path.join(result_dir, "confusion_matrix.pdf"), bbox_inches="tight")