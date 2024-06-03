from abc import ABC
from enum import Flag, auto
from time import gmtime, strftime

import matplotlib.pyplot as plt
import numpy as np


def distance(x, y):
    return np.linalg.norm(x - y, axis=2)


def closest(w, x):
    distances = distance(w, x)

    d = distances.argmin()
    i = d // distances.shape[1]
    j = d % distances.shape[1]

    return np.array([i, j])


def slow(p_lr, p_t, p_lambda):
    return p_lr * np.exp(-p_t / p_lambda)


class INBFunction(ABC):
    @staticmethod
    def nb(x, t, proximity):
        pass


class NBFunction:
    class Type(Flag):
        GAUSS = auto()
        MEXICAN_HAT = auto()

    @staticmethod
    def from_type(p_type):
        match p_type:
            case NBFunction.Type.GAUSS:
                return NBFunction.Gaussian()
            case NBFunction.Type.MEXICAN_HAT:
                return NBFunction.MexicanHat()
            case _:
                raise Exception(f'Type {p_type} not supported')

    class Gaussian(INBFunction):
        @staticmethod
        def nb(x, t, proximity):
            return np.exp(-np.power(x * t * proximity, 2))

    class MexicanHat(INBFunction):
        @staticmethod
        def nb(x, t, proximity):
            return (2 - 4 * np.power(x * t * proximity, 2)) * np.exp(-np.power(x * t * proximity, 2))


class IBoard(ABC):
    @staticmethod
    def get(M, N):
        pass


class Board:
    class Type(Flag):
        RECTANGULAR = auto()
        HEXAGONAL = auto()

    @staticmethod
    def from_type(p_type):
        match p_type:
            case Board.Type.RECTANGULAR:
                return Board.Rectangle()
            case Board.Type.HEXAGONAL:
                return Board.Hexagonal()
            case _:
                raise Exception(f'Type {p_type} not supported')

    class Rectangle(IBoard):
        @staticmethod
        def get(p_rows, p_cols):
            return np.array([
                [[j, i] for i in range(p_cols)] for j in range(p_rows)
            ])

    class Hexagonal(IBoard):
        @staticmethod
        def get(p_rows, p_cols):
            board = Board.Rectangle().get(p_rows, p_cols)

            board = board * np.array([1, np.sqrt(3) / 2])
            board = board + np.array([
                [[0.5 * (i % 2), 0] for i in range(p_cols)] for j in range(p_rows)
            ])

            return board


class KohonenNetwork:
    def __init__(self, p_rows, p_cols, p_data, nb_function, nb_proximity, board):
        self.rows = p_rows
        self.cols = p_cols
        self.dim = p_data.shape[1]

        self.nb_function = NBFunction.from_type(nb_function)
        self.nb_proximity = nb_proximity
        self.indexes = Board.from_type(board).get(p_rows, p_cols)

        # weights - positions of neurons in (self.dim)-dimensional space
        self.weights = np.random.uniform(0, 1, (p_rows, p_cols, self.dim))

        return

    def train(self, p_x, p_epochs, p_lr, p_shuffle):
        for epoch in range(p_epochs):
            indexes = list(range(len(p_x)))
            if p_shuffle:
                np.random.shuffle(indexes)

            for index in indexes:
                distances = np.linalg.norm(closest(self.weights, p_x[index]) - self.indexes, axis=2, keepdims=True)
                self.weights += (
                        self.nb_function.nb(distances, epoch, self.nb_proximity) * slow(p_lr, epoch, p_epochs) * (
                        p_x[index] - self.weights))

        return

    def to_neurons_plane(self, pos):
        return self.indexes['xs'][pos], self.indexes['ys'][pos]

    def predict(self, X):
        return np.array([distance(self.weights, x).argmin() for x in X])

    def illustrate_board(self, suf):
        plt.figure(figsize=(10, 10))
        plt.scatter(self.indexes[:, :, 0], self.indexes[:, :, 1])
        margin = 0.1
        bound = max((self.cols - 1) + 0.5, (self.rows - 1) * np.sqrt(3) / 2)
        plt.xlim((0 - margin, bound + margin))
        plt.ylim((0 - margin, bound + margin))
        plt.savefig(f'../figures/{suf}-{strftime('%H_%M_%S', gmtime())}.png')
        plt.close()

        return
