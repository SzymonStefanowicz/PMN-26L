import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

def plot_confusion_matrix(y_true, y_pred):
    """Rysuje macierz pomyłek."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Macierz Pomyłek')
    plt.ylabel('Prawdziwe etykiety')
    plt.xlabel('Przewidziane etykiety')
    plt.savefig("/Users/szymon/PycharmProjects/semestr2/PMiNAI/matrix")
    plt.show()

def plot_roc_curve(y_true, y_score):
    """Rysuje krzywą ROC."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Krzywa ROC - Regresja Logistyczna')
    plt.legend(loc="lower right")
    plt.savefig("/Users/szymon/PycharmProjects/semestr2/PMiNAI/krzywa")
    plt.show()