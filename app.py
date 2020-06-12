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



def run_the_app(gpa_w, gpa_val, hook_val, competitive, ap_val, college_group_val):
    # To make Streamlit fast, st.cache allows us to reuse computation across runs.
    # In this common pattern, we download data from an endpoint only once.
    @st.cache(allow_output_mutation=True)
    def load_data():
        df = pd.read_csv('College_numerics.csv')
        return df

    data = load_data()

    # Cleanup
    data.loc[data['gpas_uw']>4,'gpas_uw'] = data['gpas_uw'].mean()
    data["ivy_admit"] = data["ivy_admit"].astype('int')
    data["stanford_admit"] = data["stanford_admit"].astype('int')
    data["top_25_admit"] = data["top_25_admit"].astype('int')

    data['gpas_w'] = 4*data['gpas_w']/np.ceil(data['gpas_w'])
    data.dropna(subset=['gpas_w'],inplace=True)

    # Drop features
    data.dropna(subset=['sat_score'],inplace=True)

     # Pie chart
    if college_group_val == 0:
        y = data['ivy_admit']
    elif college_group_val == 1:
        y = data['stanford_admit']
    else:
        y = data['top_25_admit']
  
    #y = data['top_25_admit']
    data_sel = data.copy()


    #data_sel.drop(['gpas_uw'],axis=1,inplace=True)
    data_sel.drop(['act_score'],axis=1,inplace=True)
    data_sel.drop(['sat_score'],axis=1,inplace=True)
    data_sel.drop(['stanford_admit'],axis=1,inplace=True)
    data_sel.drop(['ivy_admit'],axis=1,inplace=True)
    data_sel.drop(['top_25_admit'],axis=1,inplace=True)
    data_sel.drop(['gpas_max'],axis=1,inplace=True)
    data_sel.drop(['Unnamed: 0'],axis=1,inplace=True)
    #data_sel.drop(['gpas_w'],axis=1,inplace=True)


    #data_dummies = data_dummies[['total_pages_visited']]

    #data_dummies['new_user'] = data_dummies['new_user'].astype(object)
    categorical_columns = [c for c in data_sel.columns 
                           if data_sel[c].dtype.name == 'object']
    numerical_columns = [c for c in data_sel.columns 
                         if data_sel[c].dtype.name != 'object']

    print('categorical:', categorical_columns)
    print('numerical:', numerical_columns)
    data_sel.dropna(inplace=True)

    if numerical_columns != [] and categorical_columns != []:
      data_new = pd.concat([data_sel[numerical_columns],
        pd.get_dummies(data_sel[categorical_columns])], axis=1)
    else:
      data_new = data_sel
    #df_median_imputed = data_new.fillna(data_new.mean())


    # Build model

    # Split to training and validation sets
    X_train, X_holdout, y_train, y_holdout = train_test_split(data_new.values, y, test_size=0.2,
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
    arr = np.array([gpa_w, gpa_val, hook_val, competitive, ap_val])
    temp = arr.reshape(1, -1)

    bb = logreg.predict_proba(temp)

    # Pie chart
    if college_group_val == 0:
        labels = ['Other colleges', 'IVY league']
    elif college_group_val == 1:
        labels = ['Other colleges', 'Stanford']
    else:
        labels = ['Other colleges', 'Top 25']

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
    else:
        return 1

def college_group_map(c_str):
    if c_str == "Ivy League":
        return 0
    elif c_str == "Stanford University":
        return 1
    else:
        return 2



def main():
    # Inputs
    st.title('Chances of getting into a Top University')
    #sat_score = st.selectbox("What is your total SAT I score?: ", np.arange(800, 1610, 10))
    gpa_val = st.text_input('What is your unweighted GPA', 3.5)
    gpa_w_val = st.text_input('What is your weighted GPA', 4.1)
    ap_val = st.text_input('How many AP classes have you taken', 2)
    first_gen_str = st.selectbox("Are you going to be the first in your family to attend college?", ["Yes", "No"])
    legacy_str = st.selectbox("Are you a legacy admit?", ["Yes", "No"])
    #STEM_str = st.selectbox("Are you planning to yolo major in STEM?", ["Yes", "No"])
    competitive_str = st.selectbox("How competitive/rigorous is your high school?",
                                   ["Not at all competitive", "Moderately competitive", "Highly competitive"])
    college_group_str = st.selectbox("Which college group chances would you like to see?", ["Ivy League", "Stanford University", "Top 25"])
    first_gen_val = 1 if first_gen_str == "Yes" else 0
    legacy_val = 1 if legacy_str == "Yes" else 0
    hook_val = first_gen_val or legacy_val
    #stem_val = 1 if STEM_str == "Yes" else 0
    competitive_val = competitive_map(competitive_str)
    college_group_val = college_group_map(college_group_str)
    run_the_app(np.float64(gpa_w_val), np.float64(gpa_val), hook_val, competitive_val, np.float64(ap_val), college_group_val)


if __name__ == "__main__":
    main()
