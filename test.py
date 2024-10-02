import math
import random

import numpy as np

w = [0, 0, 0]


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
    trainingdata = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0]]
    expresults = [0, 0, 1, 0]
    for i in range(len(w)):
        w[i] = random.uniform(0, 1.05)
    train(trainingdata, expresults)
    new = [1, 0, 0]
    print(activate(rightProp(new)))


if __name__ == "__main__":
    main()

