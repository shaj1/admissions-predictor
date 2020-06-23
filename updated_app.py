# Import libraries for data analysis
import numpy as np
import pandas as pd
import time

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
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import coo_matrix, hstack, vstack
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                          train_test_split)
import re


def run_individual_colleges(gpa_val, income_val, gender_val, hook_val, ethnicity_val, txt, college_group_val):
    
    @st.cache(allow_output_mutation=True)
    def load_data(college_val):
        if college_val == 2:
            data = pd.read_csv('Stanford_CC_results.csv')
        if college_val == 3:
            data = pd.read_csv('Duke_CC_results.csv')
        if college_val == 4:
            data = pd.read_csv('JHU_CC_results.csv')
        else:
            data = pd.read_csv('Berkeley_CC_results.csv')
        return data

    college_probs = []

    for c_val in range(2, 6):

        cc_res = load_data(c_val)
        num_of_rows = cc_res.shape[0]

        accepted=[]
        nils = 0
        positions = []
        pos_notnil = [[], [], []]
        txt = ''

        for i in range(num_of_rows):
            str1 = cc_res["body"].iloc[i]
            accepted.append(float('NaN'))
            if type(str1) == type('12') and re.search(r'(?i)decision: accepted',str1):
              accepted[i] = 1
            else:
              accepted[i] = 0


        ec=[]
        nils = 0
        positions = []
        pos_notnil = [[], [], []]

        for i in range(num_of_rows):
            str1 = cc_res["body"].iloc[i]
            ec.append('NaN')
            if type(str1) == type('12'):
              #EC = re.search(r'(?s)(?i)(?<=leadership in parenthesis\)\:).*?(?=Job/Work Experience)[^,.]',str1)
              EC = re.search(r'(?s)(?i)(?<=leadership in parenthesis\)\:).*?(?=Essays)[^,.]',str1)
            if EC and len(EC.group()) > 3 and re.search(r'(?i)decision: accepted',str1):
              ec[i] = EC.group()
              txt = txt + EC.group() + '\n'
           # if EC and len(EC.group()) > 3 and re.search(r'(?i)decision: rejected',str1):
           #   txt = txt + EC.group() + '\n'


        ec_texts = pd.DataFrame(({'texts': ec, 'accepted': accepted}))
        gpas=[]
        nils = 0
        positions = []
        pos_notnil = [[], [], []]
        gpa_pos=[]
        for i in range(num_of_rows):
            str1 = cc_res["body"].iloc[i]
            gpas.append(float("NaN"))
            if type(str1) == type('12'):
                results = re.split(r"\n+", str1)
                p = []
                for result in results:
                        #if 'gpa' in result or 'GPA' in result:
                         #   gpa_pos.append(i) 
                    if 'unweighted gpa' in result.lower():
                        gpa_nums = re.findall(r"[\d]+[\.]+[\d]{1,3}", result)
                        g = [float(gp) for gp in gpa_nums if re.match(r'^-?\d+(?:\.\d+)?$', gp) is not None and float(gp) < 100]
                        if g != []:
                            gpas[i] = min(g)

        hooks=[]
        nils = 0
        positions = []
        pos_notnil = [[], [], []]

        for i in range(num_of_rows):
            str1 = cc_res["body"].iloc[i]
            hooks.append(0)
            if type(str1) == type('12'):
              results = re.split(r"\n+", str1)
              for result in results:
                result_lc = result.lower()
                if 'hooks' in result_lc and ':' in result_lc:
                  hook_str = ','.join(re.findall(r'(?<=: ).*',result_lc))
                  if 'urm' in hook_str or 'immigrant' in hook_str  or 'first' in hook_str or 'legacy' in hook_str or 'research' in hook_str:
                    hooks[i] = 1
                elif 'hooks' in str1 and re.search('(?i)(legacy|first gen|urm|research)',str1):
                  hooks[i] = 1

        sex=[]
        positions = []
        pos_notnil = [[], [], []]

        for i in range(num_of_rows):
            str1 = cc_res["body"].iloc[i]
            sex.append(0)
            if type(str1) == type('12'):
              results = re.split(r"\n+", str1)
              for result in results:
                result_lc = result.lower()
                if 'gender:' in result_lc and 'f' in result_lc:
                  sex[i] = 1

        eth=[]
        nils = 0
        positions = []
        pos_notnil = [[], [], []]

        for i in range(num_of_rows):
            str1 = cc_res["body"].iloc[i]
            eth.append(float('NaN'))
            if type(str1) == type('12'):
              results = re.split(r"\n+", str1)
              for result in results:
                result_lc = result.lower()
                if 'ethnicity' in result_lc:
                  asian = re.findall(r"(?i)(asian|indian|wasian|azn)",result_lc)
                  white = re.findall(r"(?i)(white|w|middle)",result_lc)
                  african = re.findall(r"(?i)(african|black)",result_lc)
                  if asian != []: 
                    eth[i] = 'asian'
                  elif white != []:
                    eth[i] = 'white'
                  elif african != []:
                    eth[i] = 'african'
                  else:
                    eth[i] = 'other'

        inc=[]
        nils = 0
        positions = []
        pos_notnil = [[], [], []]

        for i in range(num_of_rows):
            str1 = cc_res["body"].iloc[i]
            inc.append(float('NaN'))
            if type(str1) == type('12'):
              results = re.split(r"\n+", str1)
              for result in results:
                result_lc = result.lower()
                if 'income bracket' in result_lc:
                  income_low = re.findall(r"\D[0-9]{2}\D",result_lc)
                  income_high = re.findall(r"\D[0-9]{3}\D",result_lc)
                  if income_low != [] or '<' in ','.join(income_low) or 'low' in ','.join(income_low):
                    inc[i] = 'low'
                  elif income_high != [] or '>' in ','.join(income_high) or 'high' in ','.join(income_high):
                    inc[i] = 'high'
                  else:
                    inc[i] = 'med'

        # Text Features (Awards)
        y_text = ec_texts['accepted']

        xtrain, xvalid, ytrain, yvalid = train_test_split(ec_texts['texts'].values, y_text, 
                                                      stratify=y_text, 
                                                      random_state=42, 
                                                      test_size=0.2, shuffle=True)

        # Always start with these features. They work (almost) everytime!
        tfv = TfidfVectorizer(min_df=3,  max_features=None, 
                    strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
                    ngram_range=(0,3), use_idf=1,smooth_idf=1,sublinear_tf=1,
                    stop_words = 'english')

        # Fitting TF-IDF to both training and test sets (semi-supervised learning)
        tfv.fit(list(xtrain) + list(xvalid))
        xtrain_tfv =  tfv.transform(xtrain) 
        xvalid_tfv = tfv.transform(xvalid)
        X = vstack([xtrain_tfv, xvalid_tfv])

        # Fitting a simple Logistic Regression on TFIDF
        clf = LogisticRegression()
        clf.fit(xtrain_tfv, ytrain)
        y_pred = clf.predict(xvalid_tfv)
        predictions = clf.predict_proba(xvalid_tfv)

        [fpr, tpr, thr] = roc_curve(yvalid, predictions[:,1])
        # get the best threshold
        J = tpr - fpr
        ix = np.argmax(J)
        best_thresh = thr[ix]
        np.unique(y_pred)
        preds = np.where(predictions[:,1] > best_thresh, 1, 0)

        cv1 = StratifiedKFold(3)
        pred_cv = cross_val_predict(clf,X,y_text,method='predict_proba')

        txt_to_doc = [txt]
        text_val = clf.predict_proba(tfv.transform(txt_to_doc))[:, 1]

        df = pd.DataFrame({'gpa_uw': gpas,'Income': inc, 'Sex': sex, 'Hooks': hooks,'Accepted':accepted, 'Ethnicity':eth, 'text_vals': pred_cv[:,1]})
        df.dropna(inplace=True)

        #df.dropna(subset=['Income'],inplace=True)
        y = df['Accepted']
        data_sel = df.copy()


        #data_sel.drop(['gpa_uw'],axis=1,inplace=True)
        data_sel.drop(['Accepted'],axis=1,inplace=True)

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

        X_train, X_holdout, y_train, y_holdout = train_test_split(data_new.values, y, test_size=0.2,
                                                                  random_state=17)

        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_holdout)
        y_pred_proba = logreg.predict_proba(X_holdout)[:, 1]
        [fpr, tpr, thr] = roc_curve(y_holdout, y_pred_proba)

        inc_high = 1 if income_val == "high" else 0
        inc_low = 1 if income_val == "low" else 0
        inc_med = 1 if income_val == "med" else 0

        eth_af = 1 if ethnicity_val == "african" else 0
        eth_as = 1 if ethnicity_val == "asian" else 0
        eth_ot = 1 if ethnicity_val == "other" else 0
        eth_wh = 1 if ethnicity_val == "white" else 0


        # Test
        arr = np.array([gpa_val, gender_val, hook_val, text_val, inc_high, inc_low, inc_med, eth_af, eth_as, eth_ot, eth_wh])
        temp = arr.reshape(1, -1)

        bb = logreg.predict_proba(temp)

        college_probs.append(bb[0][1] * 100)

        if c_val == college_group_val:
            latest_iteration = st.empty()
            bar = st.progress(0)

            for i in range(int(bb[0][1] * 100)):
              # Update the progress bar with each iteration.
              latest_iteration.text(f'Your chances: {i+1}%')
              bar.progress(i + 1)
              time.sleep(0.1)

    # BAR PLOT BEGIN

    # set font
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Helvetica'

    # set the style of the axes and the text color
    plt.rcParams['axes.edgecolor']='#333F4B'
    plt.rcParams['axes.linewidth']=0.8
    plt.rcParams['xtick.color']='#333F4B'
    plt.rcParams['ytick.color']='#333F4B'
    plt.rcParams['text.color']='#333F4B'

    # create some fake data
    percentages = pd.Series(college_probs, 
                            index=['Stanford', 'Duke', 'JHU', 'Berkeley'])
    df = pd.DataFrame({'percentage' : percentages})
    df = df.sort_values(by='percentage')

    # we first need a numeric placeholder for the y axis
    my_range=list(range(1,len(df.index)+1))

    fig, ax = plt.subplots(figsize=(5,3.8))

    # create for each expense type an horizontal line that starts at x = 0 with the length 
    # represented by the specific expense percentage value.
    plt.hlines(y=my_range, xmin=0, xmax=df['percentage'], color='#007ACC', alpha=0.2, linewidth=5)

    # create for each expense type a dot at the level of the expense percentage value
    plt.plot(df['percentage'], my_range, "o", markersize=5, color='#007ACC', alpha=0.6)

    # set labels
    ax.set_xlabel('Odds of getting in', fontsize=15, fontweight='black', color = '#333F4B')
    ax.set_ylabel('')

    # set axis
    ax.tick_params(axis='both', which='major', labelsize=9)
    plt.yticks(my_range, df.index)

    # add an horizonal label for the y axis 
    fig.text(-0.002, 0.9, 'Your Best Chances', fontsize=15, fontweight='black', color = '#333F4B')

    # change the style of the axis spines
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)

    # set the spines position
    ax.spines['bottom'].set_position(('axes', -0.04))
    ax.spines['left'].set_position(('axes', 0.015))

    ax.set_facecolor('white')

    #plt.grid(color='r')

    st.pyplot()

   # plt.savefig('hist2.png', dpi=300, bbox_inches='tight')

    # BAR PLOT END


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

    # add an horizonal label for the y axis 
    fig1, ax1 = plt.subplots(figsize=(1, 2))
    fig1.text(-0.002, 0.9, 'Your Chances', fontsize=15, fontweight='black', color = '#333F4B')
    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(int(bb[0][1] * 100)):
      # Update the progress bar with each iteration.
      latest_iteration.text(f'{i+1}%')
      bar.progress(i + 1)
      time.sleep(0.1)




    # sizes = [bb[0][0] * 100, bb[0][1] * 100]
    # # only "explode" the 2nd slice (i.e. 'Hogs')
    # explode = (0, 0.1)
    # # add colors
    # colors = ['#ff9999', '#66b3ff']
    # fig1, ax1 = plt.subplots()
    # ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
    #         shadow=True, startangle=90)
    # # Equal aspect ratio ensures that pie is drawn as a circle
    # ax1.axis('equal')
    # plt.tight_layout()
    # if st.checkbox('View Results'):
    #     st.pyplot()


def competitive_map(c_str):
    if c_str == "Not at all competitive":
        return 0
    else:
        return 1

def college_group_map(c_str):
    if c_str == "Ivy League":
        return 0
    elif c_str == "Stanford University":
        return 2
    elif c_str == "Duke University":
        return 3
    elif c_str == "Johns Hopkins University":
        return 4
    elif c_str == "University of California, Berkeley":
        return 5
    else:
        return 1

def ethinicity_group_map(c_str):
    if c_str == "Asian-American":
        return "asian"
    elif c_str == "Caucasian":
        return "white"
    elif c_str == "African-American":
        return "african"
    else:
        return "other"

def income_bracket_group_map(c_str):
    if c_str == "<50k":
        return "low"
    elif c_str == ">150k":
        return "high"
    else:
        return "med"



def main():
    # Inputs
    st.title('Chances of getting into a Top University')
    st.markdown('Provide your application information in the fields below and we\'ll tell you your odds of getting in!')
    #sat_score = st.selectbox("What is your total SAT I score?: ", np.arange(800, 1610, 10))
    gpa_val = st.sidebar.text_input('What is your unweighted GPA', 3.5)
    gpa_w_val = st.sidebar.text_input('What is your weighted GPA', 4.1)
    ap_val = st.sidebar.text_input('How many AP classes have you taken', 2)
    first_gen_str = st.sidebar.selectbox("Are you going to be the first in your family to attend college?", ["No", "Yes"])
    legacy_str = st.sidebar.selectbox("Are you a legacy admit?", ["No", "Yes"])
    gender_str = st.sidebar.selectbox("Please indicate your gender", ["Male", "Female"])
    ethnicity_str = st.sidebar.selectbox("Please indicate your Ethnicity", ["Asian-American", "Caucasian", "African-American", "Mixed"])
    income_str = st.sidebar.selectbox("Please indicate your household income bracket", ["<50k", "50k-150k", ">150k"])
    ec_aw_str = st.sidebar.text_input('Write any extracurriculars or awards obtained', 'None')
    #STEM_str = st.selectbox("Are you planning to yolo major in STEM?", ["Yes", "No"])
    competitive_str = st.sidebar.selectbox("How competitive/rigorous is your high school?",
                                   ["Not at all competitive", "Moderately competitive", "Highly competitive"])
    college_group_str = st.sidebar.selectbox("Which college group chances would you like to see?", ["Ivy League", "Stanford University", "Duke University", "Johns Hopkins University", "University of California, Berkeley", "Top 25"])
    first_gen_val = 1 if first_gen_str == "Yes" else 0
    legacy_val = 1 if legacy_str == "Yes" else 0
    hook_val = first_gen_val or legacy_val
    #stem_val = 1 if STEM_str == "Yes" else 0
    competitive_val = competitive_map(competitive_str)
    college_group_val = college_group_map(college_group_str)

    ethinicity_group_val = ethinicity_group_map(ethnicity_str)
    income_bracket_group_val = income_bracket_group_map(income_str)
    gender_group_val = 1 if gender_str == "Female" else 0

    if college_group_val > 1:
        run_individual_colleges(np.float64(gpa_val), income_bracket_group_val, gender_group_val, hook_val, ethinicity_group_val, ec_aw_str, college_group_val)
    else:
        run_the_app(np.float64(gpa_w_val), np.float64(gpa_val), hook_val, competitive_val, np.float64(ap_val), college_group_val)


if __name__ == "__main__":
    main()
