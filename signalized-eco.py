import gym
import numpy as np
from args import get_arguments
from copy import deepcopy

args = get_arguments()


class SigEcoEnv(gym.Env):
    # metadata =  {
    #     'video.frames_per_second': 2
    # }

    def __init__(self):
        self.TIME_STAMP = 0
        self.EndFlag = False
        self.RedFlag = False
        self.traffic_signals = self.Traffic_signals()
        self.Agent_EV = Agent_EV()
        self.state_new = [0]*args.STATE_DIM
        self.state = [0]*9
        self.sequence_id = [0]*args.cosd_sign_num
        self._get_state()



    def step(self, action):
        return self.state, reward, done, {}

    def reset(self):
        return self.state

    def render(self, mode='human'):
        return None

    def close(self):
        return None


    def _get_state(self):           # update env.state, env.sequence_id, env.RedFlag
        self.state[0] = np.float(deepcopy(self.Agent_EV.veh_state.v))
        self.sequence_id =self._get_signal_sequence()
        for i in range(args.cosd_sign_num):
            _id = self.sequence_id[i]
            if _id == -1:
                self.state[1+args.SIGNL_DIM*i] = float(-args.max_interval)
                self.state[2+args.SIGNL_DIM*i] = int(-args.max_cycle)
                self.state[3+args.SIGNL_DIM*i] = float(-args.max_red)
                self.state[4+args.SIGNL_DIM*i] = -1
            else:
                self.state[1+args.SIGNL_DIM*i] = float(self.traffic_signals.pos[_id] - self.Agent_EV.veh_state.dist_from_start)
                self.state[2+args.SIGNL_DIM*i] = int(self.traffic_signals.phase[_id][0])
                self.state[3+args.SIGNL_DIM*i] = float(self.traffic_signals.phase[_id][1])
                self.state[4+args.SIGNL_DIM*i] = float(np.mod((self.TIME_STAMP+self.traffic_signals.phase[_id][0]*self.traffic_signals.phase[_id][2]),self.traffic_signals.phase[_id][0])/self.traffic_signals.phase[_id][0])
            i+=1
        self.state_new[0] = np.float(deepcopy(self.Agent_EV.veh_state.v))
        self.state_new[1] = np.float(deepcopy(self.Agent_EV.veh_state.dist_from_start))




    class Traffic_signals():
        def __init__(self):
            self.id = list(range(args.signal_num))
            self.pos = self.generate_position()
            self.phase = self.generate_signal_phase()
            self.file_name = "/Env_traffic_signanl.txt"

        def generate_position(self):
            signal_pos = []
            traj_length = 0
            for i in range(args.signal_num):
                # _clip = np.random.uniform(100, 500)
                _clip = np.random.uniform(100, 200)
                signal_pos += [_clip+traj_length]
                traj_length += _clip
            return signal_pos

        def generate_signal_phase(self):
            # signal phase: [CycleTime(s), RedPhase(%), StartPhase(%)]
            signal_phase = []
            for i in range(args.signal_num):
                signal_phase += [[np.random.randint(80,161), np.random.uniform(0.3,0.7), np.random.uniform(0,1)]]
            return signal_phase

        def save(self, PATH):
            f = open(PATH+self.file_name, "w+", encoding='utf-8')
            f.write(str(self.id))
            f.write("\n")
            f.write(str(self.pos))
            f.write("\n")
            f.write(str(self.phase))
            f.close()

        def manual_set(self):
            self.pos = [346.260000000000,574.635000000000,1011.05132701422,1746.17000000000,2193.25000000000,3499.95500000000,3843.57500000000,4640.36500000000,5248.54000000000]
            CycleTime = [100]*9
            RedDuration = list(np.array([32,50,42,54,53,62,54,58,54])/100)
            StartPhase = list((200+np.array([-128.500000000000,-147.500000000000,-140.500000000000,-150.500000000000,-151.500000000000,-117.500000000000,-150.500000000000,-156.500000000000,-152.500000000000]))%100/100)
            # StartPhase = list((200+np.array([-20,-32,-72,-87,-102,-132,-117.500000000000,-150.500000000000,-156.500000000000,-152.500000000000]))%100/100)
            self.phase = [[i,j,k] for i,j,k in zip(CycleTime, RedDuration, StartPhase)]

        def manual_set_2(self):
            self.pos = [100,250,350]
            CycleTime = [100, 150, 120]
            RedDuration = (0.3, 0.5, 0.4)
            StartPhase = [0.27, 0.42, 0.28]
            self.phase = [[i,j,k] for i,j,k in zip(CycleTime, RedDuration, StartPhase)]


class Agent_EV():
        def __init__(self):
            # veh_state: [v, dist_from_start]
            # update each step:
            self.veh_state = veh_state()
            self._E_consumption = 0

        def reset(self):
            self.veh_state = veh_state()
            self._E_consumption = 0

        def update(self, action):   # Update one step action
            acc_bound = self._Tq_to_Acc_bound()
            self.action = np.clip(action, *acc_bound)
            v_c = self.veh_state.v
            self.veh_state.dist_from_start += self.veh_state.v*args.dt + 1/2*self.action*args.dt**2
            self.veh_state.v += self.action*args.dt
            _vel= np.clip(self.veh_state.v, *[0,args.max_vel])
            if self.veh_state.v != _vel:
                self.NegVFlag = True
                # print("negative v=",self.veh_state.v, "action a=", self.action)
                self.veh_state.v = _vel
            else:
                # print("action a=", self.action)
                self.NegVFlag = False
            v_n = self.veh_state.v
            self._E_consumption = self._get_E_consumption(self.action, v_c, v_n)
            if np.isnan(self.veh_state.dist_from_start) or np.isnan(self.veh_state.v):
                print("aaa")
            # Update plot @#$%


        def _Tq_to_Acc_bound(self):         # func(self.v)
            self._F_calculate()
            # _vel = np.clip(self.veh_state.v, *[0,args.max_vel])
            _Tq_max = EVeh_model.MotTqLimAtVolt(abs(self.veh_state.v)/EVeh_model.WheelRadius*EVeh_model.M2WRatio)
            acc_max = (_Tq_max/EVeh_model.WheelRadius*EVeh_model.M2WRatio-self.F)/EVeh_model.Mass
            acc_min = (-_Tq_max/EVeh_model.WheelRadius*EVeh_model.M2WRatio-self.F)/EVeh_model.Mass
            acc_bound = [acc_min, acc_max]
            return acc_bound

        def _F_calculate(self):         # func(self.v)
            self.F = EVeh_model.Fterms[0] + EVeh_model.Fterms[1]*self.veh_state.v + EVeh_model.Fterms[2]*self.veh_state.v**2

        def _get_E_consumption(self, acc, spd_c, spd_n):
            # ??? ratios ??
            T_Nm = abs(self.F + EVeh_model.Mass*acc)*EVeh_model.WheelRadius/EVeh_model.M2WRatio
            W_c_radps = spd_c/EVeh_model.WheelRadius*EVeh_model.M2WRatio
            W_n_radps = spd_n/EVeh_model.WheelRadius*EVeh_model.M2WRatio
            MECH_Pwr_c_W = T_Nm*W_c_radps
            MECH_Pwr_n_W = T_Nm*W_n_radps
            LOSS_Pwr_c_W = EVeh_model.PwLossAtVolt(W_c_radps, T_Nm)
            LOSS_Pwr_n_W = EVeh_model.PwLossAtVolt(W_n_radps, T_Nm)
            Pwr_c_W = MECH_Pwr_c_W + LOSS_Pwr_c_W
            Pwr_n_W = MECH_Pwr_n_W + LOSS_Pwr_n_W
            Pbatt_c_W = Pwr_c_W + EVeh_model.Aux_load
            Pbatt_n_W = Pwr_n_W + EVeh_model.Aux_load

            I_c = (EVeh_model.VOC_norm-np.sqrt(EVeh_model.VOC_norm**2-4*Pbatt_c_W*EVeh_model.Rin_norm))/(2*EVeh_model.Rin_norm)
            I_n = (EVeh_model.VOC_norm-np.sqrt(EVeh_model.VOC_norm**2-4*Pbatt_n_W*EVeh_model.Rin_norm))/(2*EVeh_model.Rin_norm)
            E_consumption_J = 0.5*EVeh_model.VOC_norm*(I_c+I_n)*args.dt
            if E_consumption_J<0 or np.isnan(E_consumption_J):
                print("aho")

            return E_consumption_J


class veh_state():
            def __init__(self):
                self.v = 18 + 4 * np.random.rand()
                # self.dist_from_start = 10 * np.random.rand()
                # self.v = args.max_vel*np.random.rand()
                # self.v = 20
                self.dist_from_start = 0
