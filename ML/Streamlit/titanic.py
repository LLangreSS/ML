import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier,StackingClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import streamlit as st
from xgboost import XGBClassifier
from PIL import Image

image=Image.open('ML/Streamlit/DATA/titanic.jpg')
st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Demo Titanic",
        page_icon=image,

    )
@st.cache_resource
def load():
    train_df=pd.read_csv('ML/Streamlit/DATA/train.csv')
    drop_cols=['Name','Ticket','Cabin','Fare']
    train_df.drop(drop_cols,axis=1,inplace=True)

    train_df.set_index('PassengerId',inplace=True)

    train_df['Age'].fillna(train_df['Age'].median(),inplace=True)
    train_df['Embarked'].fillna(train_df['Embarked'].mode()[0],inplace=True)

    X_train=pd.get_dummies(train_df.drop('Survived',axis=1))
    y_train=train_df['Survived']

    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

    scaler=StandardScaler()
    X_tr_scaled=scaler.fit_transform(X_tr)
    X_val_scaled=scaler.transform(X_val)

    base_model=[
        ('xgb',XGBClassifier(eval_metric='logloss')),
        ('rf',RandomForestClassifier())
         ]

    meta_model=LogisticRegression()

    model=StackingClassifier(estimators=base_model,final_estimator=meta_model)

    grid_params={
        'xgb__learning_rate': np.logspace(-3,0,10),
        'xgb__n_estimators': [50, 100, 200],
        'xgb__max_depth': [2, 3, 4,5],
        'rf__n_estimators': [50,100, 200],
        'rf__max_depth': [2,3,4,5],
        'final_estimator__C': [0.001,0.01,0.1,0.5,1,2,5,10]}

    final_model=RandomizedSearchCV(model,grid_params,cv=3,n_iter=20,n_jobs=-1)
    final_model.fit(X_tr_scaled,y_tr)
    val_pred=final_model.predict(X_val_scaled)
    accuracy=accuracy_score(y_val,val_pred)
    return final_model,X_train,accuracy,scaler,train_df

final_model,X_train,accuracy,scaler,train_df=load()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Анализ выживаемости на Титанике', fontsize=16, weight='bold')

sns.histplot(data=train_df, x='Age', hue='Survived', kde=True, ax=axes[0,0], bins=20)
axes[0,0].set_title('Распределение возраста\n(0 = погиб, 1 = выжил)')
axes[0,0].grid(True, alpha=0.3)

sns.countplot(data=train_df, x='Sex', hue='Survived', ax=axes[0,1])
axes[0,1].set_title('Выживаемость по полу')
axes[0,1].set_xlabel('')
axes[0,1].legend(['Погиб', 'Выжил'])

sns.countplot(data=train_df, x='Pclass', hue='Survived', ax=axes[1,0])
axes[1,0].set_title('Выживаемость по классу каюты')
axes[1,0].set_xlabel('Класс')
axes[1,0].legend(['Погиб', 'Выжил'])

sns.countplot(data=train_df, x='Embarked', hue='Survived', ax=axes[1,1])
axes[1,1].set_title('Выживаемость по порту посадки')
axes[1,1].set_xlabel('Порт (C=Шербур, Q=Квинстаун, S=Саутгемптон)')
axes[1,1].legend(['Погиб', 'Выжил'])

plt.tight_layout()
st.pyplot(fig)

st.write(
        """
        # Классификация пассажиров титаника
        Определяем, кто из пассажиров выживет, а кто – нет.
        """
    )
st.image(image)


sex = st.sidebar.selectbox("Пол", ("Мужской", "Женский"))
pclass = st.sidebar.selectbox("Класс", ("Первый", "Второй", "Третий"))
age = st.sidebar.slider("Возраст", min_value=1, max_value=80, value=20,
                            step=1)
embarked = st.sidebar.selectbox("Порт посадки", (
    "Шербур-Октевиль", "Квинстаун", "Саутгемптон"))
sib_sp = st.sidebar.slider(
        "Количетсво ваших братьев / сестер / супругов на борту",
        min_value=0, max_value=10, value=0, step=1)
par_ch = st.sidebar.slider("Количетсво ваших детей / родителей на борту",
                               min_value=0, max_value=10, value=0, step=1)

translatetion = {
        "Мужской": "male",
        "Женский": "female",
        "Шербур-Октевиль": "C",
        "Квинстаун": "Q",
        "Саутгемптон": "S",
        "Первый": 1,
        "Второй": 2,
        "Третий": 3,
    }

data = {
        "Pclass": translatetion[pclass],
        "Sex": translatetion[sex],
        "Age": age,
        "SibSp": sib_sp,
        "Parch": par_ch,
        "Embarked": translatetion[embarked]
    }

df=pd.DataFrame(data,index=[0])
df_dummies=pd.get_dummies(df)
for col in X_train.columns.tolist():
    if col not in df_dummies.columns.tolist():
        df_dummies[col]=0
df_dummies=df_dummies[X_train.columns.tolist()]
user_df=scaler.transform(df_dummies)
predictions=final_model.predict(user_df)
pred_proba=final_model.predict_proba(user_df)

encode_prediction = {
        0: "Сожалеем, вам не повезло",
        1: "Ура! Вы будете жить"
    }

st.write('## Точность модели')
st.write(accuracy)
predictions=encode_prediction[predictions[0]]
st.write("## Ваши данные")
st.write(df)

st.write("## Предсказание")
st.write(predictions)

st.write("## Вероятность выживания")
st.progress(float(pred_proba[0][1]))
st.write(f"{pred_proba[0][1]:.2%}")
