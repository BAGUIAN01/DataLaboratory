from unicodedata import numeric
import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import tree
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pickle

dataframe = pd.read_csv("Classeur11.csv")
numeric_column = dataframe.select_dtypes(exclude=("object")).columns
st.title("Classification")
col1,col2= st.columns(2)
model = col1.selectbox("Model",["--Choose Model--","SVC","Tree","KNN","RandomFrorest","Voting","Bagging"])
# col3,2col2= st.columns(2)
# gridsearch = col3.checkbox("GridSearch")
# randomsearch = col4.checkbox("RandomSearch")
option = col2.radio("Navigation",["Auto","RandomSearch"],horizontal=True)

colf,colt= st.columns(2)
st.sidebar.title("Select features")
feature = st.sidebar.multiselect("Features",numeric_column)
st.sidebar.title("Select target")
target = st.sidebar.selectbox("target",numeric_column)
x = dataframe[feature]
y = dataframe[target]
colx,coly= st.columns(2)
st.sidebar.title("Choose Test size")
test_size = st.sidebar.slider("Test size",min_value=1,max_value=5,help="Choose the test size")
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=test_size/10)


##############################metric###############################

def metrics(data,model):
    # x = data[feature]
    # y = data[target]
    
    # svc = SVC(C= 0.1, gamma=1, kernel='linear')
    model.fit(x_train, y_train)
    if len(y.unique()) > 2:
        col1,col2,col3,col4 = st.columns(4)
        y_pred_test = model.predict(x_test.values)
        y_pred_train = model.predict(x_train.values)
        # test
        precis_test = precision_score(y_test, y_pred_test, average='micro')
        rappel_test = recall_score(y_test, y_pred_test, average='micro')
        F1_test = f1_score(y_test, y_pred_test, average='micro')
        accur_test = accuracy_score(y_test, y_pred_test)
        # train
        precis_train = precision_score(y_train, y_pred_train, average='micro')
        rappel_train = recall_score(y_train, y_pred_train, average='micro')
        F1_train = f1_score(y_train, y_pred_train, average='micro')
        accur_train = accuracy_score(y_train, y_pred_train)
        # metrics
        col1.metric(label="Accuracy", value=round(accur_test, 3),delta=round(accur_test - accur_train, 3))
        col2.metric(label="F1 score", value=round(F1_test, 3),delta=round(F1_test - F1_train, 3))
        col3.metric(label="Recall", value=round(rappel_test, 3),delta=round(rappel_test - rappel_train, 3))
        col4.metric(label="Precision", value=round(precis_test, 3),delta=round(precis_test - precis_train, 3))
        st_learning_curve(model,x_train,y_train)

##############################st_learning_curvec###############################
def st_learning_curve(model,x_train,y_train):
    st.write("##")
    st.markdown(
        '<p class="section">Learning curves</p>',
        unsafe_allow_html=True)
    st.write("##")
    N, train_score, val_score = learning_curve(model, x_train, y_train,train_sizes=np.linspace(0.2,1.0,10),cv=3, random_state=4)
    fig = go.Figure()
    fig.add_scatter(x=N, y=train_score.mean(axis=1), name='train',
                    marker=dict(color='deepskyblue'))
    fig.add_scatter(x=N, y=val_score.mean(axis=1), name='validation',
                    marker=dict(color='red'))
    fig.update_layout(
        showlegend=True,
        template='simple_white',
        font=dict(size=10),
        autosize=False,
        width=500, height=300,
        margin=dict(l=40, r=50, b=40, t=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig)

##############################st_svc_hypperparams###############################
def st_svc_hypperparams():
    svc = SVC(C= 0.1, gamma=1, kernel='linear')
    st.header("SVC Hyper_Params")
    namec,firstc,midlec=st.columns(3)
    Min_value_C = firstc.number_input("Min_value C")
    Max_value_C = midlec.number_input("Max_value C")
    namec.header("C:")
    namegamma,firstgamma,midlegamma=st.columns(3)
    Min_value_g = firstgamma.number_input("Min_value Gamma")
    Max_value_g = midlegamma.number_input("Max_value Gamma")
    namegamma.header("GAMMA:")
        
    name,last=st.columns(2)
    name.header("Kernel :")
    kernel = last.multiselect("Choose Kernel(s)",["sigmoid","poly","rbf"])
    v1,v2 = st.columns(2)

    valider = v1.checkbox("Valider")
    
    save = v2.button("Save model")
    
    if valider:
        params = {"C":np.arange(Min_value_C,Max_value_C,0.1),"gamma":np.arange(Min_value_g,Max_value_g,0.1),"kernel":kernel}
        grid = GridSearchCV(svc,params)
        grid.fit(x_train,y_train)
        grid.best_params_
        svc = grid.best_estimator_
        metrics(dataframe,svc)
    if save:
        filename = "svc"
        pickle.dump(svc,open(filename,"wb"))
        st.balloons()


def st_KNN_hypperparams():
   
    mknn =  KNeighborsClassifier(n_neighbors=1)
    mknn.fit(x_train, y_train)
    st.header("KNN Params")
    kcl1,kcl2,kcl3=st.columns(3)
    
    kcl1.header("n_neighbors")
    k=kcl2.number_input("choose min neighbors")
    kmax=kcl3.number_input("choose max neighbors")
    wcl1,wcl2=st.columns(2)
    wcl1.header("weight")
    l= wcl2.multiselect("Choose weight",["uniform","distance"])
    acl1,acl2=st.columns(2)
    
    acl1.header("    Algorithm")
    ac=acl2.multiselect("Choose algo",["ball_tree","auto","kd_tree"])
    v1,v2 = st.columns(2)

    valider = v1.checkbox("Valider")
    
    save = v2.button("Save model")
    
    if valider:
        params = {"n_neighbors":np.arange(int(k),int(kmax),1),"weights":l,"algorithm":ac}
        grid = GridSearchCV(mknn,params)
        grid.fit(x_train,y_train)
        grid.best_params_
        mknn = grid.best_estimator_
        metrics(dataframe,mknn)
    if save:
        filename = "mknn"
        pickle.dump(mknn,open(filename,"wb"))
        st.balloons()


def st_Tree_hypperparams():
    
    mtree = tree.DecisionTreeClassifier(ccp_alpha=0.1, max_depth=3, min_samples_split=2, min_weight_fraction_leaf=0.1)
    mtree.fit(x_train, y_train)

    st.header("Tree Params")
    tcl1,tcl2,tcl3=st.columns(3)
    
    tcl1.header("min_impurity")
    t=tcl2.number_input("choose min value decrease")
    tmax=tcl3.number_input("choose max  value decrease")
    trl1,trcl2,trcl3=st.columns(3)
    trl1.header("min-leaf ")
    trs= trcl2.number_input("choose min value ")
    tres= trcl3.number_input("choose max value ")
    trls1,trcls2,trcls3=st.columns(3)
    trls1.header("min_split")
    tr= trcls2.number_input("choose min value split ")
    tre= trcls3.number_input("choose max value split ")
    tr1,tr2,tr3=st.columns(3)
    tr1.header("ccp alpha")
    trf= tr2.number_input("choose min value ccp ")
    trf2= tr3.number_input("choose max value ccp")
    tri1,tri2=st.columns(2)
    tri1.header("max features")
    trfi= tri2.multiselect("Choose one ",["auto", "sqrt", "log2","None"] )
    v1,v2 = st.columns(2)

    valider = v1.checkbox("Valider")
    
    save = v2.button("Save model")
    
    if valider:
        params = {"min_impurity_decrease":np.arange(t,tmax,0.1),"min_samples_split":np.arange(tr,tre,0.1),"min_samples_leaf":np.arange(trs,tres,0.1),"ccp_alpha":np.arange(trf,trf2,0.1),"max_features":trfi}
        grid = GridSearchCV(mtree,params)
        grid.fit(x_train,y_train)
        grid.best_params_
        mtree = grid.best_estimator_
        metrics(dataframe,mtree)
    if save:
        filename = "mtree"
        pickle.dump(mtree,open(filename,"wb"))
        st.balloons()


def st_RandomForest_hypperparams():
    raf = RandomForestClassifier(n_estimators=100)
    raf.fit(x_train, y_train)

    st.header("RandomFrorest hyperParams")
    tcl1,tcl2,tcl3=st.columns(3)
    
    tcl1.header("n_estimators")
    ta=tcl2.number_input("choose min value ",min_value=10, max_value=100, value=10)
    tmaxa=tcl3.number_input("choose max  value ",min_value=10, max_value=100, value=20)
    trl1,trcl2,trcl3=st.columns(3)
    # trl1.header("max_depth")
    # trsa= trcl2.number_input("choose mini value ")
    # tresa= trcl3.number_input("choose maxi value ")
    trls1,trcls2,trcls3=st.columns(3)
    trls1.header("min_split")
    tra= trcls2.number_input("choose min value split ")
    trea= trcls3.number_input("choose max value split ")
    tr1,tr2,tr3=st.columns(3)
    tr1.header("ccp alpha")
    trfa= tr2.number_input("choose min value ccp ")
    trf2a= tr3.number_input("choose max value ccp")
    tri1,tri2=st.columns(2)
    tri1.header("max features")
    trfia= tri2.multiselect("Choose one ",["auto", "sqrt", "log2","None"] )
    v1,v2 = st.columns(2)

    valider = v1.checkbox("Valider")
    
    save = v2.button("Save model")
    
    if valider:
        params = {"n_estimators":np.arange(int(ta),int(tmaxa)),"min_samples_leaf":np.arange(tra,trea,0.1),"ccp_alpha":np.arange(trfa,trf2a,0.1),"max_features":trfia}
        grid = GridSearchCV(raf,params)
        grid.fit(x_train,y_train)
        grid.best_params_
        raf = grid.best_estimator_
        metrics(dataframe,raf)
    if save:
        filename = "raf"
        pickle.dump(raf,open(filename,"wb"))
        st.balloons()



 
svc = SVC()
raf = RandomForestClassifier(n_estimators=100)
mknn = KNeighborsClassifier(n_neighbors=1)
mtree = tree.DecisionTreeClassifier(ccp_alpha=0.1, max_depth=3, min_samples_split=2, min_weight_fraction_leaf=0.1)
if option=="Auto" and model=="Tree":
    metrics(dataframe,svc)

elif option=="Auto" and model=="KNN":
    metrics(dataframe,mknn)

elif option=="Auto" and model=="SVC":
    metrics(dataframe,mknn)

elif option=="Auto" and model=="RandomFrorest":
    metrics(dataframe,raf)


if option=="RandomSearch" and model=="Tree":
    st_Tree_hypperparams()

elif option=="RandomSearch" and model=="SVC":
    st_svc_hypperparams()

elif option=="RandomSearch" and model=="KNN":
    st_KNN_hypperparams()

elif option=="RandomSearch" and model=="RandomFrorest":
    st_RandomForest_hypperparams()