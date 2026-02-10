import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

X = np.array([
    [30, 70],
    [25, 80],
    [27, 60],
    [31, 65],
    [23, 85],
    [28, 75]
])

y = np.array([0, 1, 0, 0, 1, 1])

label_map = {0: "Sunny", 1: "Rainy"}
colors = {0: "orange", 1: "blue"}

st.title("Weather Classifier with KNN")

st.sidebar.header("Model Settings")
n_neighbors = st.sidebar.slider("Number of Neighbors", 1, 6, 3)

knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(X, y)

temp = st.sidebar.number_input("Temperature", min_value=20, max_value=40, value=26)
humidity = st.sidebar.number_input("Humidity", min_value=50, max_value=90, value=78)

new_weather = np.array([[temp, humidity]])
pred = knn.predict(new_weather)[0]

fig, ax = plt.subplots()
ax.scatter(X[y == 0, 0], X[y == 0, 1], color=colors[0], label="Sunny", s=100, edgecolor="k")
ax.scatter(X[y == 1, 0], X[y == 1, 1], color=colors[1], label="Rainy", s=100, edgecolor="k")
ax.scatter(new_weather[0, 0], new_weather[0, 1], color=colors[pred], marker="*", s=300, edgecolor="black", label=f'New Day: {label_map[pred]}')
ax.text(new_weather[0, 0] + 0.5, new_weather[0, 1], f'Predicted: {label_map[pred]}', fontsize=12, color=colors[pred])
ax.set_xlabel("Temperature")
ax.set_ylabel("Humidity")
ax.legend()
st.pyplot(fig)

st.write(f"Prediction: {label_map[pred]}")
