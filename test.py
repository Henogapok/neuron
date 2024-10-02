import math
import random
from sklearn.datasets import load_iris
import numpy as np

w = [0, 0, 0, 0]


# Функция активации
def activate(a):
    #return 0 if a <= 0 else 1
    #return 0 if a < 0 else a
    return 1/(1+np.exp(-a))


#Функция для вычисления выхода сети
def rightProp(inputs):
    res = 0
    for i in range(len(inputs)):
        res += w[i] * inputs[i]
    return res


#Функция тренировки, подбор весов
def train(data, exp, LR=0.1, EPOCH=50):
    for i in range(EPOCH):
        for j in range(len(data)):
            error = exp[j] - activate(rightProp(data[j]))
            for x in range(len(w)):
                w[x] += LR * error * data[j][x]


def main():
    iris = load_iris()
    trainingdata = iris['data']
    expresults = iris['target']

    binary_target = np.where(expresults == 0, 0, 1)

    for i in range(len(w)):
        w[i] = random.uniform(0, 1.05)

    train(trainingdata, binary_target)
    new = [5.1, 3.5, 1.4, 0.2]
    prediction = activate(rightProp(new))
    print(prediction)

    if prediction < 0.5:
        print("Это Setosa")
    else:
        print("Это не Setosa")


if __name__ == "__main__":
    main()

