# Credit_Card_Fraud_Classification

## Introduction

### Aperçu
Ce projet se concentre sur la classification des transactions par carte de crédit afin d'identifier les activités frauduleuses. L'ensemble de données utilisé contient une répartition déséquilibrée entre les transactions normales et frauduleuses, ce qui rend essentiel l'utilisation de mesures d'évaluation appropriées.

### Dataset
L'ensemble de données utilisé pour ce projet est chargé à partir d'un fichier CSV nommé « creditcard.csv » (link: "https://www.kaggle.com/datasets/arockiaselciaa/creditcardcsv"). Il comprend des fonctionnalités telles que le temps de transaction (Time), le montant (Amount) et d'autres variables anonymisées. La variable cible « Classe » indique si une transaction est frauduleuse (Classe 1) ou normale (Classe 0).

## Model Development

### Logistic Regression
Un modèle de régression logistique est formé sur l'ensemble de données pour la détection de la fraude.

      from sklearn.linear_model import LogisticRegression
  
      lr = LogisticRegression(max_iter=500)
      lr.fit(X_train, y_train)

## Model Evaluation
Précision : capacité du modèle à prédire correctement les deux classes.
Précision : proportion de cas de fraude correctement prédits parmi tous les cas de fraude prédits.
Rappel : la proportion de cas de fraude correctement prédits parmi tous les cas de fraude réels.
Score F1 : La moyenne harmonique de la précision et du rappel.

### Les métriques sont calculées à l'aide de l'ensemble de test :

#### Predict on the test set
    y_pred = lr.predict(X_test)
#### Evaluate classifier metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

### Confusion Matrix
La matrice de confusion représente visuellement les performances du modèle :
Vrais positifs (TP) : cas de fraude correctement prédits.
Vrais négatifs (TN) : cas normaux correctement prédits.
Faux positifs (FP) : cas de fraude mal prédits.
Faux négatifs (FN) : cas normaux mal prédits.

#### Display confusion matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cmd = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred, labels=lr.classes_), display_labels=lr.classes_)
    cmd.plot()

## Conclusion
Bien que le modèle de régression logistique montre des résultats prometteurs, des ajustements supplémentaires tenant compte du déséquilibre des classes pourraient améliorer ses performances, notamment en termes de rappel pour la détection des fraudes.
