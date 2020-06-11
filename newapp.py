# Import libraries for data analysis
import numpy as np
import pandas as pd

pd.set_option("display.precision", 2)
import streamlit as st
# Data Visualization

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss



def run_the_app(sat, gpa, competitive, hook_val, stem_val):
    # To make Streamlit fast, st.cache allows us to reuse computation across runs.
    # In this common pattern, we download data from an endpoint only once.
    @st.cache(allow_output_mutation=True)
    def load_data():
        df = pd.read_csv('College_Results_MVP_updated.csv')
        return df

    df = load_data()
    # Cleanup
    df.loc[df['ivy_admit'] == 2, 'ivy_admit'] = 0
    df.loc[df['gpa_uw'] > 4, 'gpa_uw'] = 3.9

    # Drop features
    data_dummies = df.copy()
    data_dummies.drop(['ivy_admit'], axis=1, inplace=True)
    data_dummies.drop(['sat_2_score'], axis=1, inplace=True)
    data_dummies.drop(['act_score'], axis=1, inplace=True)
    data_dummies.drop(['gpa_w'], axis=1, inplace=True)
    data_dummies.drop(['gpa_max'], axis=1, inplace=True)
    data_dummies.drop(['class_rank'], axis=1, inplace=True)
    data_dummies.drop(['class_total'], axis=1, inplace=True)
    data_dummies.drop(['state_us'], axis=1, inplace=True)
    data_dummies.drop(['mit_admit'], axis=1, inplace=True)
    data_dummies.drop(['stanford_admit'], axis=1, inplace=True)
    data_dummies.drop(['best_admit'], axis=1, inplace=True)
    data_dummies.drop(['cell_number'], axis=1, inplace=True)
    data_dummies.drop(['date_posted'], axis=1, inplace=True)
    data_dummies.drop(['gender'], axis=1, inplace=True)
    data_dummies.drop(['race'], axis=1, inplace=True)
    data_dummies.drop(['t10_admit'], axis=1, inplace=True)
    data_dummies.drop(['income_bracket'], axis=1, inplace=True)

    categorical_columns = [c for c in data_dummies.columns
                           if data_dummies[c].dtype.name == 'object']
    numerical_columns = [c for c in data_dummies.columns
                         if data_dummies[c].dtype.name != 'object']

    y = df['ivy_admit']

    # Create dummy variables
    if numerical_columns != [] and categorical_columns != []:
        data_new = pd.concat([data_dummies[numerical_columns],
                              pd.get_dummies(data_dummies[categorical_columns])], axis=1)
    else:
        data_new = data_dummies

    # Impute GPA

    df_median_imputed = data_new.fillna(data_new.median())

    # Build model

    # Split to training and validation sets
    X_train, X_holdout, y_train, y_holdout = train_test_split(df_median_imputed.values, y, test_size=0.2,
                                                              random_state=17)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_holdout_scaled = scaler.transform(X_holdout)

    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_holdout)
    y_pred_proba = logreg.predict_proba(X_holdout)[:, 1]
    [fpr, tpr, thr] = roc_curve(y_holdout, y_pred_proba)

    # Test
    arr = np.array([sat, gpa, competitive, hook_val, stem_val])
    temp = arr.reshape(1, -1)

    bb = logreg.predict_proba(temp)

    # Pie chart
    labels = ['Other colleges', 'IVY league']
    sizes = [bb[0][0] * 100, bb[0][1] * 100]
    # only "explode" the 2nd slice (i.e. 'Hogs')
    explode = (0, 0.1)
    # add colors
    colors = ['#ff9999', '#66b3ff']
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.axis('equal')
    plt.tight_layout()
    if st.checkbox('View Results'):
        st.pyplot()


def competitive_map(c_str):
    if c_str == "Not at all competitive":
        return 0
    elif c_str == "Moderately competitive":
        return 1
    else:
        return 2


def main():
    # Inputs
    st.title('Chances of getting into an Ivy League School')
    sat_score = st.selectbox("What is your total SAT I score?: ", np.arange(800, 1610, 10))
    gpa_val = st.text_input('What is your unweighted GPA', 3.5)
    first_gen_str = st.selectbox("Are you going to be the first in your family to attend college?", ["Yes", "No"])
    legacy_str = st.selectbox("Are you a legacy admit?", ["Yes", "No"])
    STEM_str = st.selectbox("Are you planning to major in STEM?", ["Yes", "No"])
    competitive_str = st.selectbox("How competitive/rigorous is your high school?",
                                   ["Not at all competitive", "Moderately competitive", "Highly competitive"])
    first_gen_val = 1 if first_gen_str == "Yes" else 0
    legacy_val = 1 if legacy_str == "Yes" else 0
    hook_val = first_gen_val or legacy_val
    stem_val = 1 if STEM_str == "Yes" else 0
    competitive_val = competitive_map(competitive_str)
    run_the_app(sat_score, np.float64(gpa_val), competitive_val, hook_val, stem_val)


if __name__ == "__main__":
    main()
