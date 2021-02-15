import shap
from deepforest import CascadeForestClassifier
from sklearn.model_selection import train_test_split

# load JS visualization code to notebook
shap.initjs()

X, y = shap.datasets.iris()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = CascadeForestClassifier()

model.fit(X_train, y_train)

explainer = shap.TreeExplainer(model)
