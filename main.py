from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
import math
import random


# Предобработка данных
def standardisation(data):
    mu = []
    sigma = []
    for j in range(len(data[0])):
        mu.append(1 / len(data) * sum(data[:, j]))
        sigma.append(math.sqrt(1 / len(data) * sum(((data[:, j] - mu[j]) ** 2))))
    data = np.array(data)
    for j in range(len(data[0])):
        if sigma[j] != 0:
            data[:, j] = data[:, j] - mu[j]
            data[:, j] = data[:, j] / sigma[j]
    return data


# Перевод меток классов в представление one-hot encoding
def one_hot_encoding(target, classes):
    one_hot = np.zeros((len(target), classes))
    one_hot[np.arange(len(target)), target] = 1
    return one_hot


# Разделяем датасет на выборки и перемешиваем
def splitting(data, target, t, v):
    train = np.zeros((t, len(data[0])))
    validation = np.zeros((v, len(data[0])))
    train_target = np.zeros(t, dtype=int)
    validation_target = np.zeros(v, dtype=int)
    indexes = []

    for i in range(t):
        while True:
            index = random.randint(0, N - 1)
            if len(indexes) == 0:
                indexes.append(index)
                train[i] = data[index]
                train_target[i] = target[index]
            else:
                for ind in indexes:
                    if ind == index:
                        continue
                indexes.append(index)
                train[i] = data[index]
                train_target[i] = target[index]
            break

    for j in range(v):
        while True:
            index = random.randint(0, N - 1)
            if len(indexes) == 0:
                indexes.append(index)
                validation[j] = data[index]
                validation_target[j] = target[index]
            else:
                for ind in indexes:
                    if ind == index:
                        continue
                indexes.append(index)
                validation[j] = data[index]
                validation_target[j] = target[index]
            break

    return train, validation, train_target, validation_target


def softmax(z):
    exp = np.exp(z - np.max(z))
    for i in range(len(z)):
        exp[i] /= np.sum(exp[i])
    return exp


def training(data, target, learning_rate, c, iterations):
    global accuracy
    m, n = data.shape
    # Инициализация векторов весов и смещения
    w = np.random.normal(0, 0.1, (n, c))
    b = np.random.normal(0, 0.1, c)
    # One-hot encoding y
    y_hot = one_hot_encoding(target, c)
    losses = []

    with open("stats.txt", "w") as file:
        file.write("Обучение:" + '\n')

    # Градиентный спуск
    for iteration in range(iterations + 1):
        z = data @ w + b
        y_hat = softmax(z)
        w_grad = (1 / m) * np.dot(data.T, (y_hat - y_hot))
        b_grad = (1 / m) * np.sum(y_hat - y_hot)
        w = w - learning_rate * w_grad
        b = b - learning_rate * b_grad
        loss = -np.mean(np.log(y_hat[np.arange(len(target)), target]))
        losses.append(loss)
        z = data @ w + b
        y_hat = softmax(z)
        train_preds = np.argmax(y_hat, axis=1)
        accuracy.append(Accuracy(target, train_preds))
        if iteration % 50 == 0:
            string = f"Iteration: {iteration}, accuracy = {Accuracy(target, train_preds)}"
            with open("stats.txt", "a") as file:
                file.write(string + '\n')
            print(string)
    return w, b, losses


def Accuracy(y, y_hat):
    return np.sum(y == y_hat) / len(y)


def drawing(iterations, list, string):
    plt.plot(range(iterations), list, c="black", label=string)
    plt.legend(loc="upper left")
    plt.show()


# Колличество векторов в датасете
digits = load_digits()
N = len(digits["data"])

accuracy = []

# Длины выборок
train = int(N * 0.8)  # 80%
validation = int(N * 0.2)  # 20%

# Стандартизация и представление one-hot
data = standardisation(digits["data"])
# target = one_hot_encoding(digits["target"], 10)

# Выборки
Train, Validation, Train_target, Validation_target = splitting(data, digits["target"], train, validation)

# Стандартизированные данные (обучение)
w, b, loss = training(Train, Train_target, learning_rate=0.1, c=10, iterations=500)

# Валидационная выборка
z = Validation @ w + b
y_hat = softmax(z)
train_preds = np.argmax(y_hat, axis=1)
string = f"\nВалидация: accuracy = {Accuracy(Validation_target, train_preds)}"
with open("stats.txt", "a") as file:
    file.write(string + '\n')
print(string)

# Построение графиков
drawing(501, accuracy, "Точность")
drawing(501, loss, "Целевая функция")






