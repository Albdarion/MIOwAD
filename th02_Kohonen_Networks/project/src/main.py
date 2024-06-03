from time import gmtime, strftime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import v_measure_score

from src.network import KohonenNetwork, NBFunction, Board

SAVE_FIG = True


def task(filename, fig_prefix, M, N, epochs, lr, nb_function, board):
    # retrieve
    df = pd.read_csv(filename)
    print('read data')
    print(df)
    init_dim = df.shape[1]

    # show
    if SAVE_FIG:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(df['x'], df['y'], 0 if init_dim == 3 else df['z'], c=df['c'])
        ax.set_xlabel('x label')
        ax.set_ylabel('y label')
        ax.set_zlabel('z label')
        ax.set_title('present data')
        plt.savefig(f'../figures/{fig_prefix}-1-present-{strftime('%H_%M_%S', gmtime())}.png')
        plt.close()

    # network
    data = np.array(df.drop(columns='c', axis=1))

    kohonen = KohonenNetwork(M, N, data, nb_function, 1.0, board)
    kohonen.train(data, epochs, lr, True)
    df['pred'] = kohonen.predict(data)
    print('predicted data')
    print(df)

    # v-score
    print('v-measure score')
    print(v_measure_score(df['c'], df['pred']))

    # plot predicted
    if SAVE_FIG:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(df['x'], df['y'], 0 if init_dim == 3 else df['z'], c=df['pred'])
        ax.scatter(kohonen.weights[:, :, 0], kohonen.weights[:, :, 1], 0 if init_dim == 3 else kohonen.weights[:, :, 2],
                   c="red", label="centers", s=300, marker="x", linewidth=6, )
        ax.set_xlabel('x label')
        ax.set_ylabel('y label')
        ax.set_zlabel('z label')
        ax.set_title('predicted data')
        plt.savefig(f'../figures/{fig_prefix}-2-predicted-{strftime('%H_%M_%S', gmtime())}.png')
        plt.close()

    return


def tune_v_measure(filename, fig_prefix, M, N, epochs, lr, num, avg):
    v_measures = []
    xs = []

    for i in range(num + 1):
        whole = 0
        for j in range(avg):
            df = pd.read_csv(filename)
            predictors = df.drop(columns='c', axis=1)
            data = predictors.to_numpy()
            kohonen = KohonenNetwork(M, N, data, NBFunction.Type.GAUSS, 0.1 + i / num * 0.9, Board.Type.RECTANGULAR)
            kohonen.train(data, epochs, lr, True)
            df['pred'] = kohonen.predict(data)
            whole += v_measure_score(df['c'], df['pred'])
        xs.append(0.1 + i / num * 0.9)
        v_measures.append(whole / avg)

    plt.plot(xs, v_measures)
    plt.savefig(f'../figures/{fig_prefix}-v_measure-{strftime('%H_%M_%S', gmtime())}.png')
    plt.close()

    return


def mnist():
    df_train = pd.read_csv('../../data/mnist_train.csv', header=None)
    X_train = df_train.drop(columns=0, axis=1).to_numpy()
    y_train = df_train[0].to_numpy()

    df_test = pd.read_csv('../../data/mnist_test.csv', header=None)
    X_test = df_test.drop(columns=0, axis=1).to_numpy()
    y_test = df_test[0].to_numpy()

    kohonen_1 = KohonenNetwork(2, 5, X_train, NBFunction.Type.GAUSS, 1.0, Board.Type.RECTANGULAR)
    kohonen_2 = KohonenNetwork(2, 5, X_train, NBFunction.Type.GAUSS, 1.0, Board.Type.HEXAGONAL)
    kohonen_3 = KohonenNetwork(2, 5, X_train, NBFunction.Type.MEXICAN_HAT, 1.0, Board.Type.RECTANGULAR)
    kohonen_4 = KohonenNetwork(2, 5, X_train, NBFunction.Type.MEXICAN_HAT, 1.0, Board.Type.HEXAGONAL)

    kohonen_1.train(X_train, 100, 0.05, True)
    kohonen_2.train(X_train, 100, 0.05, True)
    kohonen_3.train(X_train, 100, 0.05, True)
    kohonen_4.train(X_train, 100, 0.05, True)

    y_test_pred_1 = kohonen_1.predict(X_test)
    y_test_pred_2 = kohonen_2.predict(X_test)
    y_test_pred_3 = kohonen_3.predict(X_test)
    y_test_pred_4 = kohonen_4.predict(X_test)

    print(v_measure_score(y_test.T[0], y_test_pred_1.T[0]))
    print(v_measure_score(y_test.T[0], y_test_pred_2.T[0]))
    print(v_measure_score(y_test.T[0], y_test_pred_3.T[0]))
    print(v_measure_score(y_test.T[0], y_test_pred_4.T[0]))

    return


def human():
    df_X_train = pd.read_table('../../data/X_train.txt', header=None, sep='\\s+')
    X_train = df_X_train.to_numpy()
    df_y_train = pd.read_table('../../data/y_train.txt', header=None, sep='\\s+')
    y_train = df_y_train.to_numpy()

    df_X_test = pd.read_table('../../data/X_test.txt', header=None, sep='\\s+')
    X_test = df_X_test.to_numpy()
    df_y_test = pd.read_table('../../data/y_test.txt', header=None, sep='\\s+')
    y_test = df_y_test.to_numpy()

    kohonen_1 = KohonenNetwork(2, 5, X_train, NBFunction.Type.GAUSS, 1.0, Board.Type.RECTANGULAR)
    kohonen_2 = KohonenNetwork(2, 5, X_train, NBFunction.Type.GAUSS, 1.0, Board.Type.HEXAGONAL)
    kohonen_3 = KohonenNetwork(2, 5, X_train, NBFunction.Type.MEXICAN_HAT, 1.0, Board.Type.RECTANGULAR)
    kohonen_4 = KohonenNetwork(2, 5, X_train, NBFunction.Type.MEXICAN_HAT, 1.0, Board.Type.HEXAGONAL)

    kohonen_1.train(X_train, 100, 0.05, True)
    kohonen_2.train(X_train, 100, 0.05, True)
    kohonen_3.train(X_train, 100, 0.05, True)
    kohonen_4.train(X_train, 100, 0.05, True)

    y_test_pred_1 = kohonen_1.predict(X_test)
    y_test_pred_2 = kohonen_2.predict(X_test)
    y_test_pred_3 = kohonen_3.predict(X_test)
    y_test_pred_4 = kohonen_4.predict(X_test)

    print(v_measure_score(y_test.T[0], y_test_pred_1.T[0]))
    print(v_measure_score(y_test.T[0], y_test_pred_2.T[0]))
    print(v_measure_score(y_test.T[0], y_test_pred_3.T[0]))
    print(v_measure_score(y_test.T[0], y_test_pred_4.T[0]))

    return


def main():
    net_rec = KohonenNetwork(10, 10, np.arange(10).reshape(2, 5), NBFunction.Type.GAUSS, 1.0, Board.Type.RECTANGULAR)
    net_rec.illustrate_board('board-rec')

    net_hex = KohonenNetwork(10, 10, np.arange(10).reshape(2, 5), NBFunction.Type.GAUSS, 1.0, Board.Type.HEXAGONAL)
    net_hex.illustrate_board('board-hex')

    # cube
    # proper number of clusters
    task('../../data/cube.csv', 'cube', 2, 4, 10, 0.1, NBFunction.Type.GAUSS, Board.Type.RECTANGULAR)

    # too many clusters
    task('../../data/cube.csv', 'cube-many', 5, 4, 10, 0.1, NBFunction.Type.GAUSS, Board.Type.RECTANGULAR)
    # too few clusters in different configurations
    task('../../data/cube.csv', 'cube-few-1', 2, 2, 10, 0.1, NBFunction.Type.GAUSS, Board.Type.RECTANGULAR)
    task('../../data/cube.csv', 'cube-few-2', 1, 4, 10, 0.1, NBFunction.Type.GAUSS, Board.Type.RECTANGULAR)

    # hexagon
    # proper number of clusters
    task('../../data/hexagon.csv', 'hexagon', 2, 3, 10, 0.1, NBFunction.Type.GAUSS, Board.Type.RECTANGULAR)

    # too many clusters
    task('../../data/hexagon.csv', 'hexagon-many', 3, 4, 10, 0.1, NBFunction.Type.GAUSS, Board.Type.RECTANGULAR)
    # too few clusters in different configurations
    task('../../data/hexagon.csv', 'hexagon-few-1', 2, 2, 10, 0.1, NBFunction.Type.GAUSS, Board.Type.RECTANGULAR)
    task('../../data/hexagon.csv', 'hexagon-few-2', 1, 4, 10, 0.1, NBFunction.Type.GAUSS, Board.Type.RECTANGULAR)

    # v-measure
    if SAVE_FIG:
        tune_v_measure('../../data/cube.csv', 'cube-v', 2, 4, 10, 0.1, 50, 5)
        tune_v_measure('../../data/hexagon.csv', 'hexagon-v', 2, 3, 10, 0.1, 50, 5)

    # # test different configurations
    # # MNIST
    # mnist()
    # # human sth dataset
    # human()

    return


if __name__ == '__main__':
    main()
