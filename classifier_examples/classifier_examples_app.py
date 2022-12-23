import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

def get_data(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    elif dataset_name == "Wine":
        data = datasets.load_wine()

    X = data.data
    y = data.target
    return X, y

def add_parameter_ui(classifier_name):
    params = dict()
    if classifier_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif classifier_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    elif classifier_name == "Random Forest":
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators",1 , 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators


    return params

def get_classifier(classifier_name, params):
    if classifier_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif classifier_name == "SVM":
        clf = SVC(C=params["C"])
    elif classifier_name == "Random Forest":
        clf = RandomForestClassifier(
            n_estimators=params["n_estimators"], max_depth=params["max_depth"], 
            random_state=42
            )

    return clf

st.title("Classifier Examples")


dataset_name = st.sidebar.selectbox("Dataset", ("Iris", "Breast Cancer", "Wine"))
st.write(f"You have chosen {dataset_name}")

classifier_name = st.sidebar.selectbox("Classifier", ("KNN", "SVM", "Random Forest"))

X, y = get_data(dataset_name)
st.write("## Dataset")
st.write("Shape of Data:", X.shape)
st.write("Number of Classes:", len(np.unique(y)))

params = add_parameter_ui(classifier_name)
clf = get_classifier(classifier_name, params)

# Classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
st.write("Classifer: ", classifier_name)
st.write("Accuracy: ", acc)

# Plot
st.write("## PCA")

pca = PCA(2)
X_projected = pca.fit_transform(X)
x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

st.pyplot(fig)