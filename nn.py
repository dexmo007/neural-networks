import numpy as np
from math import exp
import copy


def net(W, E):
    return np.matmul(W, E)


def fermi(n):
    return 1 / (1 + exp(-n))


def fermi_deriv(n):
    return fermi(n) * (1 - fermi(n))


def output(W, E, f_act):
    return list(map(f_act, net(W, E)))


def calc_d(O_exp, O):
    return [o_exp - o for o_exp, o in zip(O_exp, O)]


def calc_delta_for_output_layer(O_exp, O):
    return [o*(1-o)*(o_exp - o) for o_exp, o in zip(O_exp, O)]


def calc_delta_for_first_layer(Ws, E, O_exp, O, delta_output):
    return [fermi_deriv(neti)*sum([delta*Ws[1][i][j] for i, delta in enumerate(delta_output)]) for j, neti in enumerate(net(Ws[0], E))]


class NeuralNetwork:
    def __init__(self, weight_matrices, f_act):
        if len(weight_matrices) != 2:
            raise "Only 2 W expected"
        self.weight_matrices = weight_matrices
        self.f_act = f_act

    def calc(self, E):
        e = list(map(self.f_act, E))
        for W in self.weight_matrices:
            e = output(W, e, self.f_act)
        return e

    def calc_d(self, O_exp, O):
        return [o_exp - o for o_exp, o in zip(O_exp, O)]

    def calc_delta_i(self, O_exp, O):
        return [o*(1-o)*(o_exp - o) for o_exp, o in zip(O_exp, O)]

    def calc_delta_j(self, E, delta_i, netj=None):
        Ws = self.weight_matrices
        if netj is None:
            netj = net(Ws[0], E)
        return [fermi_deriv(neti)*sum([delta*Ws[1][i][j] for i, delta in enumerate(delta_i)]) for j, neti in enumerate(netj)]

    def calc_delta_k(self, E, delta_j):
        Ws = self.weight_matrices
        return [fermi_deriv(netk)*sum([delta*Ws[0][j][k] for j, delta in enumerate(delta_j)]) for k, netk in enumerate(E)]

    def calc_delta_generic(self, w, net, prev_delta):
        return [fermi_deriv(netj) * sum([deltai*w[i][j] for i, deltai in enumerate(prev_delta)]) for j, netj in enumerate(net)]

    def calc_delta_w_jk(self, n, delta_j, o):
        delta_w_jk = copy.deepcopy(self.weight_matrices[0])
        for j, wj in enumerate(self.weight_matrices[0]):
            for k in range(len(wj)):
                delta_w_jk[j][k] = n * delta_j[j] * o[k]
        return delta_w_jk

    def calc_delta_w_ij(self, n, d, O, E, oj=None):
        if not oj:
            oj = output(self.weight_matrices[0], E, fermi)
        delta_w_ij = copy.deepcopy(self.weight_matrices[1])
        for i, wi in enumerate(self.weight_matrices[1]):
            for j in range(len(wi)):
                delta_w_ij[i][j] = n * d[i] * O[i] * (1 - O[i]) * oj[j]
        return delta_w_ij

    def train(self, n, E, O_exp):
        e = list(map(self.f_act, E))
        nets = []
        o = [e]
        for W in self.weight_matrices:
            neti = net(W, e)
            nets.append(neti)
            e = list(map(self.f_act, neti))
            o.append(e)

        O = e

        deltas = [self.calc_delta_i(O_exp, O)]
        for i, neti in enumerate(reversed(nets[:-1])):
            delta = self.calc_delta_generic(
                self.weight_matrices[-i-1], neti, deltas[-1])
            deltas.append(delta)

        delta = self.calc_delta_generic(
            self.weight_matrices[0], E, deltas[-1])
        deltas.append(delta)

        delta_i = self.calc_delta_i(O_exp, O)
        delta_j = self.calc_delta_j(E, delta_i, nets[0])
        delta_k = self.calc_delta_k(E, delta_j)
        # print(deltas)
        # print([delta_i, delta_j, delta_k])
        delta_w_jk = self.calc_delta_w_jk(n, delta_j, o[0])
        d = self.calc_d(O_exp, O)
        delta_w_ij = self.calc_delta_w_ij(n, d, O, E, o[1])
        delta_ws = []
        for N, W in enumerate(self.weight_matrices):
            delta_wij = copy.deepcopy(W)
            for i, wi in enumerate(W):
                for j in range(len(wi)):
                    delta_wij[i][j] = n * deltas[-N-2][i] * o[N][j]
            delta_ws.append(delta_wij)

        # print(delta_ws)
        # print([delta_w_jk, delta_w_ij])
        wm = copy.deepcopy(self.weight_matrices)
        for N in range(len(self.weight_matrices)):
            wm[N] = np.add(wm[N], delta_ws[N])

        self.weight_matrices[0] = np.add(self.weight_matrices[0], delta_w_jk)
        self.weight_matrices[1] = np.add(self.weight_matrices[1], delta_w_ij)

        # print(wm)
        # print(self.weight_matrices)

        return [E] + nets, o, d, [delta_i, delta_j, delta_k]


E = [0.8, 0.5, 0.1]
O_exp = [0.4, 0.7, 0.2]

nn = NeuralNetwork([[
    [0.3, 0.2, 0.1],
    [0.4, 0.1, 0.1]
],
    [
    [-0.2, -0.3],
    [0.5, 0.4],
    [-0.8, -1],
]], fermi)

O = nn.calc(E)
d = calc_d(O_exp, O)

n = 0.5


nn.train(n, E, O_exp)
