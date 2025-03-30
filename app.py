import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.title('Streamlit Classifier Explorer')

st.write("""
# Compare Classifiers and Datasets
Which one performs best?
""")

dataset_name = st.sidebar.selectbox('Select Dataset', ('Iris', 'Breast Cancer', 'Wine'))
classifier_name = st.sidebar.selectbox('Select Classifier', ('KNN', 'SVM', 'Random Forest'))

@st.cache_data
def get_dataset(name):
    data = None
    if name == 'Iris':
        data = datasets.load_iris()
    elif name == 'Wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    X = data.data
    y = data.target
    return X, y

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0, 1.0, 0.01)
        params['C'] = C
    elif clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15, 5)
        params['K'] = K
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15, 10)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100, 10)
        params['n_estimators'] = n_estimators
    return params

def get_classifier(clf_name, params):
    if clf_name == 'SVM':
        return SVC(C=params['C'])
    elif clf_name == 'KNN':
        return KNeighborsClassifier(n_neighbors=params['K'])
    else:
        return RandomForestClassifier(n_estimators=params['n_estimators'], 
                                      max_depth=params['max_depth'], random_state=1234)

try:
    X, y = get_dataset(dataset_name)
    dataset_info = {
        'Iris': '150 samples, 4 features (sepal/petal length/width), 3 classes',
        'Breast Cancer': '569 samples, 30 features (tumor metrics), 2 classes',
        'Wine': '178 samples, 13 features (chemical properties), 3 classes'
    }
    st.write(f"### {dataset_name} Dataset")
    st.write(dataset_info[dataset_name])
    st.write('Shape of dataset:', X.shape)
    st.write('Number of classes:', len(np.unique(y)))

    params = add_parameter_ui(classifier_name)
    clf = get_classifier(classifier_name, params)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write(f'Classifier = {classifier_name}')
    st.write(f'Accuracy = {acc:.2f}')

    # Fixed Plotting
    pca = PCA(2)
    X_projected = pca.fit_transform(X)
    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]
    fig, ax = plt.subplots()
    scatter = ax.scatter(x1, x2, c=y, alpha=0.8, cmap='viridis')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    plt.colorbar(scatter, ax=ax)
    st.pyplot(fig)

except Exception as e:
    st.error(f"An error occurred: {e}")