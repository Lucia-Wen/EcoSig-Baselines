import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio


signal_num = 9

class Traffic_signals():
    def __init__(self):
        self.id = list(range(signal_num))
        self.pos = self.generate_position()
        self.phase = self.generate_signal_phase()
        self.file_name = "/Env_traffic_signanl.txt"

    def generate_position(self):
        signal_pos = []
        traj_length = 0
        for i in range(signal_num):
            # _clip = np.random.uniform(100, 500)
            _clip = np.random.uniform(100, 200)
            signal_pos += [_clip + traj_length]
            traj_length += _clip
        return signal_pos

    def generate_signal_phase(self):
        # signal phase: [CycleTime(s), RedPhase(%), StartPhase(%)]
        signal_phase = []
        for i in range(signal_num):
            signal_phase += [[np.random.randint(80, 161), np.random.uniform(0.3, 0.7), np.random.uniform(0, 1)]]
        return signal_phase


    def manual_set(self):
        self.pos = [346.260000000000, 574.635000000000, 1011.05132701422, 1746.17000000000, 2193.25000000000,
                    3499.95500000000, 3843.57500000000, 4640.36500000000, 5248.54000000000]
        CycleTime = [100] * 9
        RedDuration = list(np.array([32, 50, 42, 54, 53, 62, 54, 58, 54]) / 100)
        # StartPhase = list((200+np.array([-128.500000000000,-147.500000000000,-100.500000000000,-150.500000000000,-151.500000000000,-117.500000000000,-150.500000000000,-156.500000000000,-152.500000000000]))%100/100)
        StartPhase = list((200 + np.array([-128.500000000000, -147.500000000000, -140.500000000000, -150.500000000000, -151.500000000000, -117.500000000000, -150.500000000000, -156.500000000000, -152.500000000000])) % 100 / 100)
        # StartPhase = list((200+np.array([-20,-32,-72,-87,-102,-132,-117.500000000000,-150.500000000000,-156.500000000000,-152.500000000000]))%100/100)
        self.phase = [[i, j, k] for i, j, k in zip(CycleTime, RedDuration, StartPhase)]


traffic_signals = Traffic_signals()
traffic_signals.manual_set()

def plot_traf_vs_veh(dist,dt):
    traffic_signals = Traffic_signals()
    traffic_signals.manual_set()
    traf_t = np.linspace(0, dt*len(dist), len(dist))
    # traffic = np.array(Traffic_signals.pos).reshape(-1,1).repeat(len(dist), 1)
    Traffic = []
    plt.figure()
    for trl in range(signal_num):
        traf_temp = [traffic_signals.pos[trl] if traffic_signals.phase[trl][1] > round((t + traffic_signals.phase[trl][0] * traffic_signals.phase[trl][2]) %
                traffic_signals.phase[trl][0], 2) / traffic_signals.phase[trl][0] else None for t in traf_t]
        Traffic.append(traf_temp)
        A,=plt.plot(traf_t,traf_temp,'red',linestyle="-",label='Red light')
    B,=plt.plot(traf_t, dist, 'black', linestyle='-',label='Vehicle')
    plt.xlabel("time(s)")
    plt.ylabel("Position(m)")
    plt.legend(handles=[A,B], loc='upper left')
    plt.title("E_conspt=Propulsion+Aux_load(2.5kJ/s)", weight='bold')
    # Plot DP solution:
    plt.show()


def plot_dp_vs_rl(dt_dp, dt_rl):
    dp_data = scio.loadmat("RoutSimtoLu_v2.mat")
    # vel = dp_data['SimOut']['RoutSim'][0][0][0][0]['Vehicle']['Spd_mph'][0][0][0] * 1.60934 / 3.6
    # acc = dp_data['SimOut']['RoutSim'][0][0][0][0]['Vehicle']['Acc_mps2'][0][0][0]
    dp_dist = dp_data['SimOut']['RoutSim'][0][0][0][0]['Distance_m'][0]
    dp_dist = [a if a<5248.54000000000 else None for a in dp_dist]
    rl_data = scio.loadmat("./save/821n/E1-6/data.mat")
    rl_dist = rl_data['dist'][0][:-1]



    traffic_signals = Traffic_signals()
    traffic_signals.manual_set()
    dp_t = np.linspace(0, dt_dp*len(dp_dist), len(dp_dist))
    traf_t = np.linspace(0, dt_rl*(len(rl_dist)+10), dt_rl*len(rl_dist)/dt_dp)
    rl_t = np.linspace(0,dt_rl*len(rl_dist), len(rl_dist))
    # traffic = np.array(Traffic_signals.pos).reshape(-1,1).repeat(len(dist), 1)
    Traffic = []
    plt.figure()
    for trl in range(signal_num):
        traf_temp = [traffic_signals.pos[trl] if traffic_signals.phase[trl][1] > round((t + traffic_signals.phase[trl][0] * traffic_signals.phase[trl][2]) %
                traffic_signals.phase[trl][0], 2) / traffic_signals.phase[trl][0] else None for t in traf_t]
        Traffic.append(traf_temp)
        A,=plt.plot(traf_t,traf_temp,'red',linestyle="-",label='Red light')
    B,=plt.plot(dp_t, dp_dist, 'black', linestyle='-',label='Baseline DP Vehicle')
    C,=plt.plot(rl_t, rl_dist, 'blue', linestyle='-',label='RL Vehicle')

    plt.xlabel("time(s)")
    plt.ylabel("Position(m)")
    plt.legend(handles=[A,B,C], loc='upper left')
    plt.title("RL vs Baseline", weight='bold')
    # Plot DP solution:
    plt.show()


plot_dp_vs_rl(0.1, 1)