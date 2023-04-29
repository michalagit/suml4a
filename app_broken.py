# źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic)

import streamlit as st
import pickle
from datetime import datetime
startTime = datetime.now()
# import znanych nam bibliotek

import pathlib
from pathlib import Path

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

filename = "model.sv"
model = pickle.load(open(filename,'rb'))

def main():

	st.set_page_config(page_title="Zdrowie pacjentow")
	overview = st.container()
	left, right = st.columns(2)
	prediction = st.container()

	st.image("https://malgorzatarusek.pl/wp-content/uploads/2016/04/zdrowie-czym-jest-i-od-czego-zalezy-1140x660.jpg")

	with overview:
		st.title("Zdrowie pacjentów")
		objawy_slider = st.slider("Objawy", value=1, min_value=1, max_value=5)
		age_slider = st.slider("Wiek", value=1, min_value=1, max_value=90)
		choroby_slider = st.slider("Choroby", min_value=0, max_value=5)
		wzrost_slider = st.slider("Wzrost", min_value=0, max_value=210)
		leki_slider = st.slider("Leki", min_value=0, max_value=4)

	data = [[objawy_slider, age_slider, choroby_slider, wzrost_slider, leki_slider]]
	survival = model.predict(data)
	s_confidence = model.predict_proba(data)

	with prediction:
		st.subheader("Czy taka osoba przeżyłaby katastrofę?")
		st.subheader(("Tak" if survival[0] == 1 else "Nie"))
		st.write("Pewność predykcji {0:.2f} %".format(s_confidence[0][survival][0] * 100))

if __name__ == "__main__":
    main()
