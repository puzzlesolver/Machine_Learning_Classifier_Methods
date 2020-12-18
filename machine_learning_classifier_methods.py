# Marco Cabrera
# Machine Learning Classifier Methods
# Optimization Fucntions for each classifier methods were developed to find the optimal hyperparameters
# to tune each machine learning algorithm and fine the highest accuracy, precision, and recall scores.
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score

def main():
    st.title("Machine Learning Classifier Methods")
    st.sidebar.title("Select Classifier Methods and Metrics")
    st.markdown("Classifier Methods to evaluate risk of Alzheimer's disease after suffering TBI")
    st.sidebar.markdown("Tune Hyperparameters:")

    # Data file is loaded using Pandas
    def load_dataset():
        dataset = pd.read_csv('/Users/Marco/Desktop/Thesis/apo_e4.csv')
        # Label Encoder Method
        label_encoder_method = LabelEncoder()
        # For-Loop with fit_transform() method
        for variable in dataset.columns:
            dataset[variable] = label_encoder_method.fit_transform(dataset[variable].astype(str))
        return dataset

    # Split method is used to split, train, and test the data
    def split_data(dataframe):
        y = dataframe['nincds_arda_diagnosis']
        x = dataframe.drop(columns=['nincds_arda_diagnosis'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)
        return x_train, x_test, y_train, y_test

    # Find optimal C hyper-parameter for Support Vector Machine Classifier
    def find_optimal_c_svm():
        dataframe = load_dataset()
        x_train, x_test, y_train, y_test = split_data(dataframe)
        c = 0
        i = 0
        f = open("C:/Users/Marco/Desktop/TEST2/svm.txt", "a")
        accuracy_c = 0
        precision_c = 0
        recall_c = 0
        highest_accuracy = 0
        highest_precision = 0
        highest_recall = 0
        # while loop to iterate through all possible combinations
        while (c <= 100):
            c += 0.1
            i += 1
            # Support Vector Machine Hyper-parameters used: C, Kernel, and Gamma
            machine_learning_model = SVC(C=c, kernel="rbf", gamma="scale")
            machine_learning_model.fit(x_train, y_train)
            y_pred = machine_learning_model.predict(x_test)
            # Accuracy, precision, and recall scores
            accuracy = machine_learning_model.score(x_test, y_test)
            precision = precision_score(y_test, y_pred, labels=axis_legends)
            recall = recall_score(y_test, y_pred, labels=axis_legends)

            # Find the optimal combination of iterations in order to find the C hyper-parameter
            # that will generate the most optimal accuracy, precision, and recall results.
            if (accuracy > highest_accuracy):
                highest_accuracy = accuracy
                accuracy_c = c

            if (precision > highest_precision):
                highest_precision = precision
                precision_c = c

            if (recall > highest_recall):
                highest_recall = recall
                recall_c = c

            f.write(str(i) + ": " + str(accuracy) + "\n")
        # Write optimal C hyper-parameter results on text file after all possible combinations
        f.write("Optimal C for Accuracy: " + str(accuracy_c) + "\n")
        f.write("Optimal C for Precision: " + str(precision_c) + "\n")
        f.write("Optimal C for Recall: " + str(recall_c) + "\n")
        f.write("Highest Accuracy Result: " + str(highest_accuracy) + "\n")
        f.write("Highest Precision Result: " + str(highest_precision) + "\n")
        f.write("Highest Recall Result: " + str(highest_recall) + "\n")
        f.close()

    # Find optimal C hyperparameter for Logistic Regression Classifier
    def find_optimal_hyperparameters_logistic_regression():
        dataframe = load_dataset()
        x_train, x_test, y_train, y_test = split_data(dataframe)
        # define variables for optimization
        optimal_c_accuracy = 0
        optimal_c_precision = 0
        optimal_c_recall = 0
        max_num_iter_accuracy = 0
        max_num_iter_precision = 0
        max_num_iter_recall = 0
        highest_accuracy = 0
        highest_precision = 0
        highest_recall = 0
        # for-loops to iterate through all possible combinations
        for i in range(1, 100, 1):
            c = i / 100
            f = open("C:/Users/Marco/Desktop/TEST2/outputsLogisticRegression.txt", "a")
            for max_iter in range(1, 101):
                # Logistic Regression hyper-parameters used: C, penalty, and max_iter
                machine_learning_model = LogisticRegression(C=c, penalty='l2', max_iter=max_iter)
                machine_learning_model.fit(x_train, y_train)
                y_pred = machine_learning_model.predict(x_test)
                accuracy = machine_learning_model.score(x_test, y_test)
                precision = precision_score(y_test, y_pred, labels=axis_legends)
                recall = recall_score(y_test, y_pred, labels=axis_legends)

                # Find the optimal combination of iterations in order to find the C hyper-parameter
                # that will generate the most optimal accuracy, precision, and recall results.
                if (accuracy > highest_accuracy):
                    highest_accuracy = accuracy
                    optimal_c_accuracy = c
                    max_num_iter_accuracy = max_iter
                if (precision > highest_precision):
                    highest_precision = precision
                    optimal_c_precision = c
                    max_num_iter_precision = max_iter
                if (recall > highest_recall):
                    highest_recall = recall
                    optimal_c_recall = c
                    max_num_iter_recall = max_iter
            # Write optimal hyper-parameters results on text file after all possible combinations
            f.write("Optimal C for Accuracy: " + str(optimal_c_accuracy) + "\n")
            f.write("Maximum Number of Iterations for Accuracy: " + str(max_num_iter_accuracy) + "\n")
            f.write("ACCURACY: " + str(highest_accuracy) + "\n")
            f.write("Optimal C for Precision: " + str(optimal_c_precision) + "\n")
            f.write("Maximum Number of Iterations for Precision: " + str(max_num_iter_precision) + "\n")
            f.write("PRECISION: " + str(highest_precision) + "\n")
            f.write("Optimal C for Recall: " + str(optimal_c_recall) + "\n")
            f.write("Maximum Number of Iterations for Recall: " + str(max_num_iter_recall) + "\n")
            f.write("RECALL: " + str(highest_recall) + "\n")
            f.write("##############################################################################\n")
            f.close()

    # Find Optimal Combination of Height and Depth parameters for Random Forest Classifier Method
    def find_optimal_hyperparameters_random_forest():
        dataframe = load_dataset()
        x_train, x_test, y_train, y_test = split_data(dataframe)
        # define variables for optimization
        highest_accuracy = 0
        highest_precision = 0
        highest_recall = 0
        optimal_number_of_trees_accuracy = 0
        optimal_depth_accuracy = 0
        optimal_number_of_trees_precision = 0
        optimal_depth_precision = 0
        optimal_number_of_trees_recall = 0
        optimal_depth_recall = 0

        # Find optimal depth and optimal number of trees in Random Forest Classifier Method
        # The number of trees in the forest
        for n in range(10, 600, 10):
            f = open("C:/Users/Marco/Desktop/TEST2/optimalRandomForest.txt", "a")
            # The maximum depth of the tree.
            for d in range(10, 600, 10):
                # Random Forest hyper-parameters used: n_estimators, max_depth, bootstrap, and n_jobs
                machine_learning_model = RandomForestClassifier(n_estimators=n, max_depth=d, bootstrap='true', n_jobs=-1)
                machine_learning_model.fit(x_train, y_train)
                y_pred = machine_learning_model.predict(x_test)
                accuracy = machine_learning_model.score(x_test, y_test)
                precision = precision_score(y_test, y_pred, labels=axis_legends)
                recall = recall_score(y_test, y_pred, labels=axis_legends)

                # Find the optimal combination of iterations in order to find the hyper-parameters
                # that will generate the most optimal accuracy, precision, and recall results.
                if (accuracy > highest_accuracy):
                    highest_accuracy = accuracy
                    optimal_number_of_trees_accuracy = n
                    optimal_depth_accuracy = d
                if (precision > highest_precision):
                    highest_precision = precision
                    optimal_number_of_trees_precision = n
                    optimal_depth_precision = d
                if (recall > highest_recall):
                    highest_recall = recall
                    optimal_number_of_trees_recall = n
                    optimal_depth_recall = d

            # Write optimal hyper-parameter results on text file after all possible combinations
            f.write("n,d:" + str(n) + "," + str(d) + "\n")
            f.write("Optimal height accuracy: " + str(optimal_number_of_trees_accuracy) + "\n")
            f.write("Optimal depth accuracy: " + str(optimal_depth_accuracy) + "\n")
            f.write("ACCURACY: " + str(highest_accuracy) + "\n")
            f.write("Optimal height precision: " + str(optimal_number_of_trees_precision) + "\n")
            f.write("Optimal depth precision: " + str(optimal_depth_precision) + "\n")
            f.write("PRECISION: " + str(highest_precision) + "\n")
            f.write("Optimal height recall: " + str(optimal_number_of_trees_recall) + "\n")
            f.write("Optimal depth recall: " + str(optimal_depth_recall) + "\n")
            f.write("RECALL: " + str(highest_recall) + "\n")
            f.close()

    # Plot Machine Learning Classifier Methods to display results
    def plot_classification_models(performance_measurement_tools):
        if 'Confusion Matrix' in performance_measurement_tools:
            st.subheader("Confusion Matrix")
            st.subheader("A.D. = Alzheimer's Disease")
            confusion_matrix_color = 'hot'
            plot_confusion_matrix(machine_learning_model, x_test, y_test, display_labels=axis_legends, cmap=confusion_matrix_color)
            st.pyplot()

        if 'ROC Curve' in performance_measurement_tools:
            st.subheader("ROC Curve")
            roc_color = 'black'
            plot_roc_curve(machine_learning_model, x_test, y_test, color=roc_color)
            st.pyplot()

        if 'Precision-Recall Curve' in performance_measurement_tools:
            st.subheader('Precision-Recall Curve')
            precision_recall_curve_color = 'black'
            plot_precision_recall_curve(machine_learning_model, x_test, y_test, color=precision_recall_curve_color)
            st.pyplot()

    dataframe = load_dataset()
    axis_legends = ["No A.D.", "A.D."]

    find_optimal_c_svm()
    find_optimal_hyperparameters_logistic_regression()
    find_optimal_hyperparameters_random_forest()

    x_train, x_test, y_train, y_test = split_data(dataframe)

    st.sidebar.subheader("Select a Machine Learning Classifier")
    machine_learning_method = st.sidebar.selectbox("",("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))
    # Support Vector Machine (SVM) Classifier Method
    if machine_learning_method == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Regularization Parameter")
        # define optimal hyper-parameters C, Kernel, and Gamma for optimal Accuracy, Precision, and Recall results
        c_parameter = st.sidebar.number_input("C Hyperparameter", 0.01, 100.0, step=0.01, key='c_parameter')
        kernel_parameter = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel_parameter')
        gamma_parameter = st.sidebar.radio("Gamma", ("scale", "auto"), key='gamma_parameter')

        # Select Confusion Matrix, ROC Curve, Precision-Recall Curve to plot and show performance-measurement results
        performance_measurement_graph = st.sidebar.multiselect("Select Classification Model:",
                                                              ('Confusion Matrix','ROC Curve','Precision-Recall Curve'))
        # Show Accuracy, Precision, and Recall results and round up to four decimals only after defining parameters
        if st.sidebar.button("Show Results", key='train_and_test'):
            st.subheader("Support Vector Machine (SVM) Results")
            machine_learning_model = SVC(C=c_parameter, kernel=kernel_parameter, gamma=gamma_parameter)
            machine_learning_model.fit(x_train, y_train)
            y_pred = machine_learning_model.predict(x_test)
            accuracy = machine_learning_model.score(x_test, y_test)
            precision = precision_score(y_test, y_pred, labels=axis_legends)
            recall = recall_score(y_test, y_pred, labels=axis_legends)
            # Print Accuracy, Precision, and Recall results
            st.write("Accuracy: ", accuracy.round(4))
            st.write("Precision: ", precision.round(4))
            st.write("Recall: ", recall.round(4))
            # Plot Confusion Matrix, ROC Curve, Precision-Recall Curve
            plot_classification_models(performance_measurement_graph)

    # Logistic Regression Classifier Method
    if machine_learning_method == 'Logistic Regression':
        st.sidebar.subheader("Regularization Parameter")
        # define optimal hyper-parameters for optimization function
        c_parameter = st.sidebar.number_input("C (Regularization parameter)", 0.01, 100.0, step=0.01, key='c_parameter')
        maximum_iterations = st.sidebar.slider("Maximum number of iterations", 1, 100, key='maximum_iterations')
        # Select Confusion Matrix, ROC Curve, Precision-Recall Curve to plot and show performance-measurement results
        performance_measurement_graph = st.sidebar.multiselect("Select Classification Model:",
                                                              ('Confusion Matrix','ROC Curve','Precision-Recall Curve'))
        # Show Accuracy, Precision, and Recall results and round up to four decimals only after defining parameters
        if st.sidebar.button("Show Results", key='train_and_test'):
            st.subheader("Logistic Regression Results")
            machine_learning_model = LogisticRegression(C=c_parameter, penalty='l2', max_iter=maximum_iterations)
            machine_learning_model.fit(x_train, y_train)
            accuracy = machine_learning_model.score(x_test, y_test)
            y_pred = machine_learning_model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(4))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=axis_legends).round(4))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=axis_legends).round(4))
            # Plot Confusion Matrix, ROC Curve, Precision-Recall Curve
            plot_classification_models(performance_measurement_graph)

    # Random Forest Classifier Method
    if machine_learning_method == 'Random Forest':
        st.sidebar.subheader("Regularization Parameter")
        # define optimal hyper-parameters for optimization function
        number_estimators = st.sidebar.number_input("Number of trees", 10, 600, step=10, key='number_estimators')
        maximum_depth = st.sidebar.number_input("Maximum depth", 10, 200, step=1, key='maximum_depth')
        bootstrap_samples = st.sidebar.radio("Bootstrap", ('True', 'False'), key='bootstrap_samples')
        # Select Confusion Matrix, ROC Curve, Precision-Recall Curve to plot and show performance-measurement results
        performance_measurement_graph = st.sidebar.multiselect("Select Classification Model:",
                                                              ('Confusion Matrix','ROC Curve','Precision-Recall Curve'))
        # Show Accuracy, Precision, and Recall results and round up to four decimals only after defining parameters
        if st.sidebar.button("Show Results", key='train_and_test'):
            st.subheader("Random Forest Results")
            machine_learning_model = RandomForestClassifier(n_estimators=number_estimators, max_depth=maximum_depth,
                                                            bootstrap=bootstrap_samples, n_jobs=-1)
            machine_learning_model.fit(x_train, y_train)
            accuracy = machine_learning_model.score(x_test, y_test)
            y_pred = machine_learning_model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(4))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=axis_legends).round(4))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=axis_legends).round(4))
            # Plot Confusion Matrix, ROC Curve, Precision-Recall Curve
            plot_classification_models(performance_measurement_graph)

    if st.sidebar.checkbox("Show Raw Patient Dataset", False):
        st.subheader("TBI Patient Data Set (Label Encoded)")
        st.write(dataframe)

if __name__ == '__main__':
    main()
