import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(page_title='Whitney Rey Capstone Project', page_icon=":smiley_cat:", layout='centered')

@st.cache_data
def load_data():
    file_path = "ai4i2020.csv"
    df = pd.read_csv(file_path)
    return df

df = load_data()
print(df.head())
print(df.info())

# drop UDI and product ID - unnecessary to project
df = df.drop(['Product ID', 'UDI', 'Type', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)

# adapted from https://www.geeksforgeeks.org/change-data-type-for-one-or-more-columns-in-pandas-dataframe/#
df = df.astype(float)
print(df.dtypes)

# Separate independent variables for data exploration
df2 = df[['Air temperature [K]', 'Process temperature [K]',
          'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]

st.title('Exploring Machine Learning as a Solution to Predictive Maintenance in Lean Manufacturing')

# adapted from https://www.youtube.com/watch?v=LYXTKpnwB2I&t=60s
tab_titles = [
    "Raw Data",
    "Processed Data",
    "Feature Engineering",
    "Predictive Model",
    "Conclusions",
    "HELP"
]

tabs = st.tabs(tab_titles)
with tabs[0]:
    st.header('Introduction to Dataset')
    st.markdown("""
    Data exploration is critical to model creation, as it allows the developer to customize the program based on the dataset.
    Further, it allows the developer to make adjustments (such as data transformation) that allows for reduced errors in predictive models.
    The table below presents the data to be analyzed representing 10,000 observations of a manufacturing milling machine retrieved from https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
    """)
    st.write(df)
    st.header('Data Exploration')
    st.markdown("""
    The graph below provides visualization of the raw data.
    """)
    st.line_chart(df)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.markdown("""
            Initial descriptive statistics help understand basic spread of data within the dataset. Refer to the table below.
            """)
    descstats = df.describe(include='all')
    st.write(descstats)
    st.markdown("""
             Histograms help visualization data distribution, skewness, and outliers. Notice that each variable follows a different distrubtion.
             """)
    # Histogram shows data distributions
    # Adapted from https://www.statology.org/pandas-histogram-for-each-column/
    df2.hist(layout=(2, 3))
    plt.subplots_adjust(bottom=0.01, right=2, top=2)
    st.pyplot()
    st.markdown("""
                 Using a counter function of the dependent variable shows severe unbalancing of machine failure observations. 
                 Efforts to reduce error caused by unbalancing will be considered when sampling (i.e. using oversampling to split test/train sets).
                 """)
    st.write(df['Machine failure'].value_counts())

with tabs[1]:
    # Adapted from https://docs.streamlit.io/library/api-reference/widgets/st.selectbox
    st.header('Outlier Management')
    st.markdown("""
                    In order to create a successful model, data cleaning and preprocessing should be considered.
                    Common approaches to this is to conduct outlier treatment and/or normalize the data using a data transformation. 
                    Outlier management was initially explored by trimming values that fell beyond the interquantile range - about 5%
                    of observations fell beyond the IQR limits. Though trimming the dataset by 5% could be justified, further
                    investigation showed that trimming would result in s 22.5% reducition in "machine failed" dependent variable observations.
                    Since the dependent variable is already in a state of severe unbalancing, it would be problematic to further reduce
                    "machine failed" observations. Furthermore, it is likely that an extreme condition (such as rpm) would be related
                    to an observed machine failure. For these reasons, it was determined that maintaining integrity of the dataset 
                    is of greater benefit than that of outlier removal.
                    """)
    # plot adapted from https://www.geeksforgeeks.org/box-plot-in-python-using-matplotlib/#
    st.markdown("""
                 Boxplots visualize the presence of outliers as shown below.
                 """)
    plot_columns = [col for col in df2]
    df[plot_columns].plot(kind='box', figsize=(12, 6), title='Box and Whisker Plots', ylabel='Value', grid=True)
    st.pyplot()

    st.markdown("""
                       Count of dependent variable observations prior to outlier treatment:
                       """)
    st.write(df['Machine failure'].value_counts())
    # outlier management
    # adapted from: https://www.geeksforgeeks.org/interquartile-range-to-detect-outliers-in-data/
    # Timming method used adatped from https://www.analyticsvidhya.com/blog/2022/09/dealing-with-outliers-using-the-iqr-method/
    binary_variables = ['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    df_IQR = df
    for col in df_IQR.columns:
        if col not in binary_variables:
            # calculate the IQR (interquartile range)
            Q1 = df_IQR[col].quantile(0.25)
            Q3 = df_IQR[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df_IQR[(df_IQR[col] <= (Q1 - 1.5 * IQR)) | (df_IQR[col] >= (Q3 + 1.5 * IQR))]
            if not outliers.empty:
                df_IQR.drop(outliers.index, inplace=True)

    # Trimming removed 22.4% of observed machine failures.
    # This significantly alters the integrity of the dataset.
    # Because outliers may have a direct impact on machine failure, outlier management will not be preformed.

    st.markdown("""
                    Count of dependent variable observations after outlier treatment (22.5% reduction in '1' observations):
                         """)
    st.write(df_IQR['Machine failure'].value_counts())

    st.header('Data Transformation')
    # Data transformation
    # normalize data using z-score
    # Adapted from https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.zscore.html
    from scipy.stats import zscore

    binary_variables = ['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    for col in df.columns:
        if col not in binary_variables:
            df[col] = zscore(df[col])
    st.markdown("""
                    As shown in the previous section, each variable follows a different distribution. To allow for 
                    effective alignment and variable comparison, the raw data was transformed using z-score calculations.
                    The updated histograms below show significant shift in rotational speed and torque as a direct result
                    of standard normalization. 
                             """)
    # Updated histograms post cleaning and transformation
    # Adapted from https://www.statology.org/pandas-histogram-for-each-column/
    df.hist(layout=(2, 3))
    plt.subplots_adjust(bottom=0.01, right=1.5, top=2)
    plt.show()
    st.pyplot()

with tabs[2]:
    st.header('Correlation of Parameters')
    st.markdown("""
                    To best understand the impact of various independent variables on the dependent variable (Machine Failure), feature
                    correlation is conducted. Referencing the correlation matrix, there is strong correlation between process temperature and
                    environment temperature. This shows that if the system is sensitive to temperature, air controls for the manufacturing
                    floor may be worth considering. Lastly, the correlation matrix shows there is correlation between machine failure, torque, and tool wear.
                    Because there is not a clear discrepancy between independent variables and their correlation to the dependent variable,
                    all variables will be used in modeling. 
                       """)

    # Geeks for Geeks, 2020
    # Adapted from: https://www.statology.org/correlation-in-python/#:~:text=This%20tutorial%20explains%20how%20to%20calculate%20the%20correlation,we%20can%20use%20the%20Numpy%20corrcoef%20%28%29%20function.
    correlation_matrix = df.corr()

    correlation_with_failure = correlation_matrix['Machine failure']
    sorted_correlation = correlation_with_failure.sort_values(ascending=False)
    # visualization
    # Adapted from:https://datagy.io/python-correlation-matrix/
    plt.figure(figsize=(10, 5));
    map_matrix = df.corr().round(2)
    mask = np.triu(np.ones_like(map_matrix, dtype=bool))
    map = sns.heatmap(map_matrix, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag', mask=mask)
    st.pyplot()

with tabs[3]:
    # Data split using 70/30 ratio
    st.header('Test/Train Subset Creation')
    st.markdown("""
                        This section provides three proposed solutions using machine learning as a tool for predictive modeling.
                        First, the Naive Bayes algorithm is used. Second, Random Forest decision trees approach is used. Lastly, k-Nearest-Neighbors is used (KNN)

                         As shown in the raw data portion of this application, the observations of failed events and passing events
                         is very unbalanced. Having an unbalanced dependent variable may negatively impact predictive performance.
                         To address this, over sampling was used using SVMSMOTE. To ensure the data is now balanced, a histogram
                         of the dependent variable is shown below - notice observations are ratio 1:1.
                           """)
    # Data split using 70/30 ratio
    # adapted from https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    from sklearn.model_selection import train_test_split

    X = df.drop(['Machine failure'], axis=1)
    y = df["Machine failure"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    # Oversample to combat unbalanced data (Bronwlee, 2020)
    # Adapted from https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SVMSMOTE.html
    from imblearn.over_sampling import SVMSMOTE

    X_train, y_train = SVMSMOTE(random_state=50).fit_resample(X_train, y_train)

    # Check if over sampling worked
    plt.figure(figsize=(10, 5));
    plt.hist(y_train)
    st.pyplot()

    # Build Naive Bayes Model
    # Use time(time) to help track efficency of model
    # adapted from: https://www.geeksforgeeks.org/python-time-time-method/
    st.header('Naive Bayes Classifier')
    st.markdown("""
                    The Naive Bayes classifier is based on the Gaussian Naive Bayes theorem. The constructed model used
                    default settings provided in Sklearn. Presenting the following confusion matrix:
                              """)

    from sklearn.naive_bayes import GaussianNB
    import time
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report, \
        confusion_matrix, ConfusionMatrixDisplay

    # Build Model
    # Use time(time) to help track efficency of model
    # adapted from: https://www.geeksforgeeks.org/python-time-time-method/
    start = time.time()
    NBmodel = GaussianNB().fit(X_train, y_train)
    stop = time.time()
    y_pred = NBmodel.predict(X_test)

    # evaluate
    # adapted from: https://www.datacamp.com/tutorial/naive-bayes-scikit-learn
    accuracy_nb = accuracy_score(y_pred, y_test)
    f1_nb = f1_score(y_pred, y_test, average="weighted")
    time_nb = stop - start

    # adapted from: https://www.datacamp.com/tutorial/naive-bayes-scikit-learn
    labels = ["Normal Operations", "Machine Failed"]

    # adapted from: https://www.datacamp.com/tutorial/naive-bayes-scikit-learn
    plt.figure(figsize=(5, 10));
    cm_nb = confusion_matrix(y_test, y_pred)
    disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm_nb)
    st.markdown("""
                    Naive Bayes Confusion Matrix:
                    """)

    disp_nb.plot();
    st.pyplot()
    st.header('Random Forest Classifier')
    st.markdown("""
                       The Random Forest classifier utilizes many decision trees to classify an observation. 
                       A class is decided based on the majority scoring determined by the decision tree pathway.
                       The first decision tree is presented below - notice that the desicion tree begins with both a positive 
                       and negative value (0 class vs 1 class) and then branches out to various scenarios to ensure proper classification.

                                 """)
    # Random forest
    # Parameter Tuning Resource: https://www.analyticsvidhya.com/blog/2015/06/tuning-random-forest-model/#:~:text=When%20tuning%20a%20random%20forest%2C%20key%20parameters%20to,of%20a%20split%20%28e.g.%2C%20Gini%20impurity%20or%20entropy%29.
    # Adapted from: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    from sklearn.ensemble import RandomForestClassifier

    start = time.time()
    RFmodel = RandomForestClassifier(n_estimators=25, n_jobs=-1,
                                     random_state=50).fit(X_train, y_train)
    stop = time.time()
    y_pred = RFmodel.predict(X_test)
    accuracy_rf = accuracy_score(y_pred, y_test)
    f1_rf = f1_score(y_pred, y_test, average="weighted")
    time_rf = stop - start
    plt.figure(figsize=(20, 10))
    rf = confusion_matrix(y_test, y_pred)
    disp_rf = ConfusionMatrixDisplay(confusion_matrix=rf)

    from sklearn import tree
    from sklearn.tree import plot_tree

    # Adapted from https://mljar.com/blog/visualize-tree-from-random-forest/

    tree_to_visualize = RFmodel.estimators_[0]

    feature_names = ['Air temperature [K]', 'Process temperature [K]',
                     'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    # Plot the first decision tree

    tree = plot_tree(tree_to_visualize, filled=True, feature_names=feature_names, class_names=np.unique(y).astype(str))
    plt.show()
    st.pyplot()
    st.markdown("""
                    Though the decision trees map provides great visualization of the classifier's decision pathway,
                    it is difficult to read. To better visualize the exact variables assessed and weight assigned, 
                    a second tree is presented, that zooms into the first two branches of the larger tree map.
                                    """)

    # Adapted from https://mljar.com/blog/visualize-tree-from-random-forest/
    tree_graph = RFmodel.estimators_[0]

    # Set the maximum depth for better readability
    plt.figure(figsize=(25, 10))
    plot_tree(tree_graph, filled=True, feature_names=feature_names, class_names=np.unique(y).astype(str), max_depth=2)
    st.pyplot()

    st.markdown("""
                     Random Forest Confusion Matrix:
                     """)
    plt.figure(figsize=(10, 5));
    disp_rf.plot()
    st.pyplot()

    st.header('KNN Classifier')
    st.markdown("""
                    K-Nearest Neighbors relies on a predefined hyperparameter to determine classification boundaries.
                    GridSeachCV was used to determine the optimal k-value based on model accuracy metrics as shown below:
                    """)
    # KNN Model Creation
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import GridSearchCV

    # Adapted from https://dev.to/balapriya/cross-validation-and-hyperparameter-search-in-scikit-learn-a-complete-guide-5ed8
    start = time.time()
    knn_class = KNeighborsClassifier()

    # Use grid search function for hyperparamter tuning
    hyperparameter_range = {'n_neighbors': range(2, 20)}
    find_hyperpar = GridSearchCV(knn_class, hyperparameter_range, cv=5, scoring='accuracy')
    find_hyperpar.fit(X_train, y_train)
    st.write(find_hyperpar.best_params_)

    KNNmodel = KNeighborsClassifier(n_neighbors=2).fit(X_train, y_train)
    stop = time.time()
    y_pred = KNNmodel.predict(X_test)

    accuracy_knn = accuracy_score(y_pred, y_test)
    f1_knn = f1_score(y_pred, y_test, average="weighted")
    time_knn = stop - start

    knn = confusion_matrix(y_test, y_pred)
    disp_knn = ConfusionMatrixDisplay(confusion_matrix=knn)
    st.markdown("""
                       Using hyperparameter k =2 and default 5-fold cross validation, the constructed model presents the following confusion matrix.
                       """)
    st.markdown("""
                    KNN Confusion Matrix:
                    """)
    plt.figure(figsize=(10, 5));
    disp_knn.plot()
    st.pyplot()
    st.header('Model Comparison')
    # EVALUATION
    st.markdown("""
                To evaluate the model, the accuracy, f1 score, and training time (efficiency) are displayed in the table below.
                A confusion matrix is provided for each model to visualize precision.
                """)
    # comparison table
    # Adapted from https://learnpython.com/blog/print-table-in-python/
    from tabulate import tabulate

    table = {'model': ['Naive Bayes', 'Random Forest', 'KNN'], 'Accuracy': [accuracy_nb, accuracy_rf, accuracy_knn],
             'Precision': [f1_nb, f1_rf, f1_knn], 'Train Time': [time_nb, time_rf, time_knn]}
    evaluatedf = pd.DataFrame(data=table)
    print(evaluatedf)
    st.write(evaluatedf)

with tabs[4]:
    st.header('Key Conclusions')
    st.markdown("""
             Based on the evaluation methods displayed in Tab "Predictive Model" it was shown that: \n
             1. Naive Bayes provides acceptable accuracy, with decent precision and low training time. \n
             2. Random Forest presents highest accuracy and precision, while requiring relatively low training time. \n
             3. KNN provides high accuracy and precison, but significantly greater training time than other classifiers. When scaled to a larger dataset, the train time will decrease the benefit of high accuracy. \n
             \n
             4. Review of the confusion matrices show the NB classifier to have the greatest occurance of false negatives. This impacts predictive maintenance applications, as false negatives would 
             not prove an improvement to current preventative maintenance methods. Both RF and KNN models show better recall and precision, which alignes properly with predictive maintenance use-case.
             \n
             \n
             """)
    st.header('Final Recommendation')
    st.markdown(""" 
             The overall business objective is to reduce cost. Referencing the evulation metrics and confusion matrices, the RANDOM FOREST model presents the best business solution. \n
             This is because Random Forest presents high accuracy, with minimial instaces of incorrect predictions without being overly conservative and requiring unnecessary maintenance (confusion matrix). \n
             For these reasons, Random Forest would be the best option for this use case.""")

with tabs[5]:
    st.markdown("""
                        This application requires processing power - to ensure optimal results, please use the following hardware: \n
                        1. Intel Core i% processor (or equivalent) \n
                        2. 4 GB RAM \n
                        3. 15 GB available hard disk space \n
                        \n
                        This application has been tested on Windows operating systems. Future expansion of product may include IOS. \n
                        \n
                        This application is currently customized to meet the needs of UCI's Al4I 2020 Predictive Maintenance Dataset. For this reason,
                        there is no option for user-upload. \n
                        \n
                        If an element is not loading (i.e. a new graph after selecting from sidebar), please relaunch the applciation. \n
                        \n
                        The application will auto-update based upon user input from the sidebar drop down menu(s) and checkbox(s). \n
                        \n
                        For additional concerns and comments, please contact application developer at wdavis19@my.gcu.edu
                        """)
