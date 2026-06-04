import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Memuat dataset Titanic
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Memilih fitur yang relevan
df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']]

# Menangani nilai yang hilang
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Encoding variabel kategorikal
le_sex = LabelEncoder()
df['Sex'] = le_sex.fit_transform(df['Sex'])

le_embarked = LabelEncoder()
df['Embarked'] = le_embarked.fit_transform(df['Embarked'])

# Memisahkan fitur dan target
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = df['Survived']

# Membagi data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Melatih model RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Menyimpan model ke file pickle
with open('titanic_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model sudah disimpan sebagai 'titanic_model.pkl'")
#####

import streamlit as st
import pickle
import pandas as pd

# Memuat model dari pickle
with open('titanic_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Fungsi untuk melakukan prediksi
def predict_survival(pclass, sex, age, sibsp, parch, fare, embarked):
    data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked': [embarked]
    })
    
    prediction = model.predict(data)
    return prediction[0]

# Menyusun aplikasi Streamlit
st.title("Prediksi Kelangsungan Hidup Titanic")

st.write("""
    Aplikasi ini akan memprediksi apakah seorang penumpang Titanic selamat atau tidak berdasarkan beberapa fitur.
""")

# Input dari pengguna
pclass = st.selectbox('Pilih Kelas Penumpang (Pclass)', [1, 2, 3])
sex = st.radio('Jenis Kelamin', ['Laki-laki', 'Perempuan'])
sex = 1 if sex == 'Laki-laki' else 0  # Encoding: Laki-laki = 1, Perempuan = 0
age = st.slider('Usia', 0, 100, 25)
sibsp = st.slider('Jumlah Saudara/Suami di Titanic (SibSp)', 0, 10, 0)
parch = st.slider('Jumlah Orang Tua/Anak di Titanic (Parch)', 0, 10, 0)
fare = st.number_input('Tarif Tiket (Fare)', min_value=0.0, step=1.0, value=7.25)
embarked = st.selectbox('Pelabuhan Naik (Embarked)', ['C', 'Q', 'S'])
embarked = 0 if embarked == 'C' else (1 if embarked == 'Q' else 2)  # Encoding: C = 0, Q = 1, S = 2

# Tombol untuk memprediksi
if st.button('Prediksi'):
    prediction = predict_survival(pclass, sex, age, sibsp, parch, fare, embarked)
    if prediction == 1:
        st.success("Penumpang Diprediksi Selamat!")
    else:
        st.error("Penumpang Diprediksi Tidak Selamat.")
