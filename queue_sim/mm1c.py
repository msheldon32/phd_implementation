import random
import sys
import math

def simulate(arriv, serv, cap, n_iter):
    cum_Q_t = 0
    cum_U_t = 0
    t = 0
    Q = 0

    for i in range(n_iter):
        if Q == 0:
            Q += 1
            t += math.log(random.random())/arriv
        elif Q == cap:
            dt = math.log(random.random())/serv
            cum_Q_t += (dt*Q)
            cum_U_t += dt
            Q -= 1
            t += dt
        elif random.random() <= (arriv/(serv+arriv)):
            dt = math.log(random.random())/(serv+arriv)
            cum_Q_t += (dt*Q)
            cum_U_t += dt
            Q += 1
            t += dt
        else:
            dt = math.log(random.random())/(serv+arriv)
            cum_Q_t += (dt*Q)
            cum_U_t += dt
            Q -= 1
            t += dt
    return [cum_Q_t/t, cum_U_t/t]

if __name__ == "__main__":
    arriv  = float(sys.argv[1])
    serv   = float(sys.argv[2])
    cap    = int(sys.argv[3])
    n_iter = int(sys.argv[4])

    print(f"mean Q length, U: {simulate(arriv, serv, cap, n_iter)}")
