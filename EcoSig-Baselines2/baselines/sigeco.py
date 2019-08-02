import gym
from gym import spaces
import numpy as np
from copy import deepcopy
from scipy.interpolate import interp2d, interp1d
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt


# args_env = get_arguments()
# kwargs = get_arguments
mod = "dim3"
class EVeh_model():
    Mass = 2.2135e+3
    Fterms = np.array([162.4046, 4.8429, 0.4936])
    _Spd_rpm = np.array([0,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,11900,12400])
    _Spd_radps = _Spd_rpm * np.pi/30
    _Tq_Nm = np.array([-415.7, -405.0, -400, -350, -300, -250, -233, -225, -200, -175, -150, -125, -100, -75, -50, -25, -10, 0, 10, 25, 50, 75, 100, 125, 150, 175, 200, 225, 233, 250, 300, 350, 400, 405.0, 415.7])
    _Pw_W = np.array([[14351.2843432000,14091.1418970700,14587.1574276255,15837.8561117835,16391.8315070421,13341.4903214869,12845.7226528034,12930.9894360770,13286.9938053932,16960.8265302376,20637.3428820956,24316.0446000527,27996.4335487945,31309.8143708543,33150.8639615047],
                    [13874.3406600000,13608.9069996700,14096.8387444255,15319.9055559835,15876.4913018421,12980.5593338869,12512.7762740034,12604.2043740770,12958.9388459932,16534.9430116376,20113.6307828956,23694.5039416527,27277.0643097945,30502.3994328543,32294.5347462824],
                    [13651.4697800000,13383.5635896700,13867.7178644255,15077.8725859835,15635.6781218421,12811.8999938869,12357.1938540034,12451.5010740770,12805.6421359932,16335.9320216376,19868.9055028956,23404.0643816527,26940.9104597945,30125.1027328543,31894.3809073935],
                    [11422.7609900000,11130.1295296700,11576.5090744255,12657.5429159835,13227.5462518421,11125.3065838869,10801.3696840034,10924.4681140770,11272.6751059932,14345.8221316376,17421.6527628956,20499.6687816527,23579.3719997945,26352.1356928543,27892.8424340602],
                    [9194.05219800000,8876.69546266998,9285.30028242546,10237.2132459835,10819.4143818421,9438.71317788691,9245.54550600336,9397.43514507695,9739.70807199319,12355.7122416376,14974.4000128956,17595.2731716527,20217.8335397945,22579.1686628543,23891.3039762824],
                    [6965.34340700000,6623.26139666999,6994.09149142546,7816.88357698345,8411.28251284211,7752.11977188691,7689.72133000336,7870.40217807695,8206.74103899319,10365.6023516376,12527.1472628956,14690.8775716527,16856.2950797945,18806.2016328543,19889.7655185047],
                    [6207.58241800000,5857.09381366999,6215.08050242546,6993.97148898345,7592.51767784210,7178.67801388691,7160.74111100336,7351.21096907695,7685.53224799319,9688.96499063756,11695.0813328956,13703.3830616527,15713.3719997945,17523.3928428543,18529.2424451713],
                    [5850.98901100000,5496.54436366999,5848.48709542546,6606.71874198345,7207.21657884210,6908.82306888691,6911.80924200336,7106.88569407695,7440.25752299319,9370.54740863756,11303.5208928956,13238.6797716527,15175.5258497945,16919.7181128543,17888.9962818380],
                    [4599.89011000000,4471.59930866999,4796.94863442546,5477.92753298345,6003.15064484210,6065.52636488691,6133.89715500336,6343.36921107695,6673.77400599319,8375.49246363756,10079.8945168956,11786.4819716527,13494.7566197945,15033.2345928543,15888.2270451713],
                    [3637.58241800000,3588.74216566999,3884.20138142546,4495.40006098345,4963.37042484210,5222.22966188691,5355.98506700336,5579.85272707695,5907.29048999319,7380.43751863756,8856.26814289557,10334.2841616527,11813.9873897945,13146.7510828543,13887.4578240602],
                    [2933.07692300000,2834.89601166999,3096.72885342546,3636.05940098345,4053.59020584210,4378.93295888691,4578.07297900336,4816.33624407695,5140.80697299319,6385.38257363756,7632.64176889557,8882.08636265270,10133.2181547945,11260.2675628543,11886.6885901713],
                    [2208.35164800000,2177.31359466999,2404.75083142546,2876.71874198345,3245.56822784210,3535.63625488691,3800.16089100336,4052.81976007695,4374.32345699319,5390.32762863756,6409.01539589557,7429.88856065270,8452.44892479447,9373.78404685427,9885.91935972688],
                    [1618.90109900000,1606.43447366999,1793.54204042546,2194.52093998345,2517.10668884210,2782.11977188691,3022.24880300336,3289.30327707695,3607.83993999319,4395.27268363756,5185.38902189557,5977.69075765270,6771.67969379447,7487.30052985427,7885.15012828243],
                    [1117.36263700000,1112.47842966999,1257.60797442546,1581.33412698345,1852.05174384210,2094.09779388691,2321.58946200336,2525.78679307695,2841.35642399319,3400.21773863756,3961.76264789557,4525.49295565270,5090.91046279447,5600.81701385427,5884.38089839354],
                    [668.901098900000,681.379528269985,788.816765925456,1036.93852198345,1254.24954584210,1460.69119988691,1671.80924200336,1858.42415607695,2074.87290699319,2405.16279363756,2738.13627489557,3073.29515365270,3410.14123179447,3714.33349785427,3883.61166850466],
                    [296.923076900000,312.588319469985,379.805776925455,544.740719983452,703.809985642103,865.856035386906,1055.21583590336,1227.65492507695,1408.05972099319,1627.36059563756,1849.34506589557,2073.51493365270,2299.37200179447,2503.67415685427,2617.45782139354],
                    [107.252747300000,129.621286469985,180.245337425455,307.268192483452,433.040754942103,573.108782686906,746.094956703356,908.314265876951,1052.67510519319,1266.70125463756,1483.41099989557,1702.30614265270,1922.88848479447,2122.44338785427,2233.58969017132],
                    [0,57.4234842899851,97.0585242054554,224.191269383452,336.667128542103,467.504387086906,645.655396303356,803.588991076951,997.290489793187,1160.87707903756,1327.14726389557,1495.60284565270,1665.74562779447,1819.90492585427,1905.83144761577],
                    [107.252747300000,147.643264469985,197.827754925455,326.169291383452,462.601194442103,602.339551886906,789.721330403356,940.182397676951,1117.51027009319,1289.99795763756,1465.16924189557,1642.52592265270,1821.56980379447,1983.74009085427,2074.11716206021],
                    [296.923076900000,336.764143669985,406.728853925455,571.883577083452,736.996798842103,904.317573886906,1086.97407760336,1240.62195807695,1423.33444599319,1609.66828763756,1798.68572489557,1989.88856065270,2182.77859479447,2357.41042085427,2454.71056928243],
                    [668.901098900000,716.324583169985,827.058524225456,1067.92753298345,1278.31547984210,1465.74614488691,1652.02902300336,1800.18239807695,1986.30147899319,2297.25070563756,2610.88352789557,2926.70174665270,3244.20716579447,3530.99283785427,3690.60067850466],
                    [1117.36263700000,1157.75315466999,1306.28929342546,1618.47698398345,1858.09569984210,2054.53735388691,2233.23781400336,2375.23734307695,2654.32345699319,3215.05290363756,3778.46594489557,4344.06438465270,4911.35002379447,5422.93789285427,5707.43584294910],
                    [1618.90109900000,1667.42348466999,1858.04753542546,2233.64181898345,2499.96383184210,2691.35054088691,2866.42462700336,3041.94063907695,3557.73004999319,4526.59136463756,5498.13627489557,6471.86658265270,7447.28408879447,8326.19063985427,8814.75452439354],
                    [2208.35164800000,2258.30260466999,2484.86072242546,2921.11434598345,3209.96383184210,3405.85603588691,3583.67737400336,3708.64393607695,4461.13664399319,5838.12982663756,7217.80660489557,8599.66877965270,9983.21815479447,11229.4433828543,11922.0731990602],
                    [2933.07692300000,2929.84106666999,3193.21237042546,3679.02643398345,3990.84295284210,4208.93295888691,4399.06199000336,4375.34723307695,5364.54323699319,7149.66828763756,8937.47693389557,10727.4709816527,12519.1522197945,14132.6961328543,15029.3918851713],
                    [3637.58241800000,3708.96194566999,4005.30028242546,4532.43302798345,4880.51328284210,5127.28460688691,5214.44660500336,5042.05052907695,6267.94983099319,8461.20674963756,10657.1472628956,12855.2731716527,15055.0862897945,17035.9488828543,18136.7105685047],
                    [4599.89011000000,4623.35755066999,4949.58599642546,5512.98247798345,5897.76602984210,6188.27361788691,6029.83122000336,5708.75382607695,7171.35642399319,9772.74521063756,12376.8175928956,14983.0753716527,17591.0203497945,19939.2016328543,21244.0292573935],
                    [5850.98901100000,5661.92897866999,6007.82775542546,6596.16929098345,7025.01877684210,7249.26262888691,6845.21583600336,6375.45712307695,8074.76301699319,11084.2836716376,14096.4879228956,17110.8775716527,20126.9544197945,22842.4543728543,24351.3479251713],
                    [6207.58241800000,6017.75315466999,6368.04753542546,6957.15830198345,7399.52427184210,7588.77911188691,7106.13891300336,6588.80217807695,8363.85312699319,11503.9759816376,14646.7824328956,17791.7742716527,20938.4533197945,23771.4952528543,25345.6899051713],
                    [6965.34340700000,6773.87952866999,7133.51456842546,7724.25995098345,8195.34844684210,8310.25163988691,7660.60045100336,7042.16042007695,8978.16961099319,12395.8221316376,15816.1582528956,19238.6797716527,22662.8884897945,25745.7071228543,27458.6666085047],
                    [9194.05219800000,8997.78062666998,9384.88819442546,9980.44126898345,10536.0077918421,10432.2296638869,9291.36968200336,8375.56701307695,10784.9827959932,15018.8990616376,19255.4989128956,23494.2841616527,27734.7566197945,31552.2126228543,33673.3039807269],
                    [11422.7609900000,11221.6817296700,11636.2618244255,12236.6225859835,12876.6671318421,12554.2076838869,10922.1389140034,9708.97360607695,12591.7959859932,17641.9759816376,22694.8395728956,27749.8885616527,32806.6247497945,37358.7181128543,39887.9413373935],
                    [13651.4697800000,13445.5828296700,13887.6354444255,14492.8039059835,15217.3264718421,14676.1857038869,12552.9081440034,11042.3802040770,14398.6091759932,20265.0529016376,26134.1802328956,32005.4929516527,37878.4928797945,43165.2236028543,46102.5786940602],
                    [13874.3406600000,13667.9729396700,14112.7728144255,14718.4220359835,15451.3924018421,14888.3835038869,12715.9850640034,11175.7208640770,14579.2904859932,20527.3605916376,26478.1142928956,32431.0533916527,38385.6796897945,43745.8741528543,46724.0424329491],
                    [14351.2843432000,14143.8877750700,14594.5667862255,15201.2448341835,15952.2934920421,15342.4867958869,13064.9696728034,11461.0698764770,14965.9484893932,21088.6990482376,27214.1331812956,33341.7527332527,39471.0594631945,44988.4663298543,48053.9748341713]])
    PwLossAtVolt = interp2d(_Spd_radps, _Tq_Nm, _Pw_W)

    _Spd_rpm_mot = np.array([0,510,1020,1530,2040,2550,3060,3570,4080,4467,4591,5101,5611,6121,6631,7141,7651,8161,8671,9182,9692,10202,10712,11222,11732,12140,12400,12500])
    _Spd_radps_mot = _Spd_rpm_mot * np.pi/30
    _Tq_Nm_mot = np.array([413.126736111111,414.711805555556,414.783854166667,415,415.144097222222,415.072048611111,415,414.783854166667,409.668402777778,400.518229166667,396.555555555556,370.401909722222,340.429687500000,312.402777777778,287.113715277778,264.418402777778,243.668402777778,225.151909722222,208.220486111111,193.810763888889,180.914062500000,170.034722222222,160.164062500000,151.518229166667,143.520833333333,137.540798611111,133.866319444444,0])
    MotTqLimAtVolt = interp1d(_Spd_radps_mot, _Tq_Nm_mot)
    Inertia = 16.4161
    M2WRatio = 9.4876
    WheelRadius = 0.3522
    Aux_load = 2000
    Rin_norm = 0.0872279723055434
    VOC_norm = 3.507459685926577e+02


signal_num = 3      #6      #3      #4      #9      |   3
max_step = 2e3      #2e3    #1e3    #5e3    #3e3    | 1e3
max_dist= 5.4e3      #4e3    #1200   #2e3    #5400   | 500
max_interval=1500    #1500   #500    #-      #1500   |   -
reward_step = 0.2

max_vel=12500*np.pi/30/EVeh_model.M2WRatio*EVeh_model.WheelRadius
dt=0.5
SIGNL_DIM=4
# STATE_DIM=2
cosd_sign_num=2
STATE_DIM=cosd_sign_num*SIGNL_DIM+1


max_cycle=180
max_red=1

E_reward=1e-4
Red_reward=max_step*dt  #1000
End_reward=max_step*dt
RedFar_reward=max_step*dt
GStop_reward=max_step*dt/5  #200
# Red_reward=200
# End_reward=50


memory_capacity=max_step*2






class SigEcoEnv(gym.Env):

    def __init__(self, **kwargs):
        self.TIME_STAMP = 0
        self.EndFlag = False
        self.RedFlag = False
        self.GreenFlag = False
        self.RedFar = False
        self.GreenStop = 0
        self.traffic_signals = Traffic_signals()
        self.traffic_signals.manual_set()
        self.Agent_EV = Agent_EV()
        self.state_new = [0]*3   #[v, dist, t]
        self.state = [0]*STATE_DIM      #[v, rel_dis, cycle, red, current]
        self.sequence_id = [0]*cosd_sign_num
        self._get_state()

        if mod == "dim3":
            self.observation_space = spaces.Box(low=np.array([0,0,0]),high=np.array([max_vel, max_dist, max_step*dt]))
        elif mod == "dim5":
            self.observation_space = spaces.Box(low=np.array([0]*STATE_DIM),high=np.array([max_vel, max_interval, max_cycle, 0.7, 1, 2000, max_cycle, 0.7, 1]))
            # self.observation_space = spaces.Box(low=np.array([0]*STATE_DIM),high=np.array([max_vel]+[max_interval, max_cycle, 0.7, 1]*cosd_sign_num))
            # self.observation_space = spaces.Box(low=np.array([0,0]),high=np.array([max_vel, max_dist]))
        self.action_space = spaces.Box(low=np.array([-10]), high=np.array([10]))
        self.viewer = None
        self._viewers={}


    def step(self, action):
        self.TIME_STAMP += dt
        self.Agent_EV.update(action)
        self._get_state()
        self._check_end()
        self._check_green_stop()
        self._check_red_dist()
        reward = self._get_reward()
        if mod == "dim5":
            state = normalize_state(self.state)
        elif mod == "dim3":
            state = normalize_state(self.state_new)
        # state = normalize_state(self.state_new)
        # state = self.state
        done = self.EndFlag or self.RedFlag or bool(self.GreenStop>=10) or self.RedFar
        return state, reward, done, {"action_real": self.Agent_EV.action}

    def reset(self):
        self._reset_agent()
        if mod == "dim5":
            return normalize_state(self.state)
        elif mod == "dim3":
            return normalize_state(self.state_new)
        # return self.state_new
        # return self.state


    def render(self, mode='human'):
        screen_width = 600
        screen_height = 20

        # if self.viewer is None:
        #     from gym.envs.classic_control import rendering
        #     self.viewer = rendering.Viewer(screen_width, screen_height)
        # self.fig_env = plt.figure("env_fig", figsize = (100, 30))
        # plt.axes(xlim=(-10, 500), ylim=(- 150, 150))

        plt.figure("env_fig")
        plt.title("Demo")
        plt.cla()

        plt.plot([-10, 500], [-3,-3], ":k")
        plt.plot([-10, 500], [3, 3], ":k")

        # Plot vehicles:
        line = plt.plot([], [], "k")[0]
        _x = self.Agent_EV.veh_state.dist_from_start
        line.set_data([_x-5,_x-5,_x+0,_x+0,_x-5], [-1.5,1.5,1.5,-1.5,-1.5])

        # Plot all traffic lights:
        for i in range(signal_num):
            if self.traffic_signals.phase[i][1]>round((self.TIME_STAMP+self.traffic_signals.phase[i][0]*self.traffic_signals.phase[i][2])%self.traffic_signals.phase[i][0],2)/self.traffic_signals.phase[i][0]:
                plt.plot(self.traffic_signals.pos[i], 0, "ro")
            else:
                plt.plot(self.traffic_signals.pos[i], 0, "go")

        plt.pause(0.001)
        print("vel = ", self.state_new[0], "    act = ", self.Agent_EV.action, "    rew = ", self._get_reward())
        # if self.state_new[0]==0:
        #     print("wierd")





    def close(self):
        if self.viewer is not None:
            self.viewer = None
            self._viewers={}


    def _get_state(self):           # update env.state, env.sequence_id, env.RedFlag
        self.state[0] = np.float(deepcopy(self.Agent_EV.veh_state.v))
        self.sequence_id =self._get_signal_sequence()
        for i in range(cosd_sign_num):
            _id = self.sequence_id[i]
            if _id == -1:
                self.state[1+SIGNL_DIM*i] = float(max_interval)
                self.state[2+SIGNL_DIM*i] = int(max_cycle)
                self.state[3+SIGNL_DIM*i] = 0
                self.state[4+SIGNL_DIM*i] = 0
            else:
                self.state[1+SIGNL_DIM*i] = float(self.traffic_signals.pos[_id] - self.Agent_EV.veh_state.dist_from_start)
                self.state[2+SIGNL_DIM*i] = int(self.traffic_signals.phase[_id][0])
                self.state[3+SIGNL_DIM*i] = float(self.traffic_signals.phase[_id][1])
                self.state[4+SIGNL_DIM*i] = float(np.mod((self.TIME_STAMP+self.traffic_signals.phase[_id][0]*self.traffic_signals.phase[_id][2]),self.traffic_signals.phase[_id][0])/self.traffic_signals.phase[_id][0])
            i+=1
        self.state_new[0] = np.float(deepcopy(self.Agent_EV.veh_state.v))
        self.state_new[1] = np.float(deepcopy(self.Agent_EV.veh_state.dist_from_start))
        self.state_new[2] = np.float(deepcopy(self.TIME_STAMP))


        # if  self.state_new[0]==0:
        #     print("wierd")

    def _check_end(self):
        if self.sequence_id[0] == -1:
            self.EndFlag = True

    def _check_green_stop(self):
        if self.state[4]>self.state[3] and self.state[0]<1:
            self.GreenStop += 1
        else:
            self.GreenStop = 0

    def _check_red_dist(self):
        if self.state[4]<self.state[3] and self.state[0]<0.01:
            if self.state[1]>5:
                self.RedFar = True
            else:
                self.RedFar = False
        else:
            self.RedFar = False

    def _get_reward(self):
        # reward(a|self._state, self.state): E_consumption, Time, Red_Flag
        self.reward_Red, self.reward_Green, self.reward_Green_Stop, \
        self.reward_End, self.reward_Pass, self.reward_Spd, self.reward_red_far = 0, 0, 0, 0, 0, 0, 0
        self.reward_E = -E_reward*self.Agent_EV._E_consumption
        if self.RedFlag:
            # self.reward_Red = -Red_reward
            self.reward_Red = -Red_reward*1#(0.5+self.state[0]/max_vel)
            self.reward_End = End_reward
        if self.TIME_STAMP>max_step*dt and not self.EndFlag:
            self.reward_End = -End_reward*(1-self.state_new[1]/max_dist)
            self.EndFlag = True
        if self.PassFlag:
            self.reward_Pass = Red_reward/5
        if bool(self.GreenStop>=10):
            self.reward_Green_Stop = -GStop_reward
        if self.RedFar:
            self.reward_red_far = -RedFar_reward*(0.5+self.state[1]/max_interval)
            print("red_far:", self.state[1])
        # reward = self.reward_E + self.reward_End + self.reward_Red - 0*T_reward + self.reward_Spd
        # reward = self.reward_E + self.reward_End + self.reward_Red+self.reward_Spd
        if mod =="dim3":
            reward = self.reward_Red - reward_step + self.reward_Pass + self.reward_Green_Stop + self.reward_red_far
        if mod =="dim5":
            reward = self.reward_Red - reward_step + self.reward_Pass + self.reward_Green_Stop + self.reward_red_far
        return reward

    def _reset_agent(self):
        self.TIME_STAMP = 0
        self.EndFlag = False
        self.RedFlag = False
        self.GreenFlag = False
        self.GreenStop = 0
        self.Agent_EV.reset()
        self.sequence_id = [0] * cosd_sign_num
        self._get_state()
        return self.state

    def _get_signal_sequence(self):     # func(self.Agent_EV.veh_state.dist_from_start)
        # update: PassFlag, RedFlag
        for i in list(range(self.sequence_id[0], signal_num)):
            if self.sequence_id[0]==signal_num-1:
                if self.Agent_EV.veh_state.dist_from_start >= self.traffic_signals.pos[i]:
                    signal_sequence=[-1]*cosd_sign_num
                    self.PassFlag=True
                    if self.state[4]<self.state[3]:
                        self.RedFlag=True
                        self.GreenFlag=False
                    else:
                        self.RedFlag=False #self.GreenFlag=True
                        self.GreenFlag=True
                    return signal_sequence
                else:
                    signal_sequence=self.sequence_id
                    self.PassFlag=False
                    self.RedFlag=False
                    self.GreenFlag=False
                    return signal_sequence
            else:
                if self.Agent_EV.veh_state.dist_from_start >= self.traffic_signals.pos[i]:
                    i += 1
                    self.PassFlag = True
                    if self.state[4]<self.state[3]:
                        self.RedFlag=True
                        self.GreenFlag=False
                    else:
                        self.RedFlag=False #self.GreenFlag=True
                        self.GreenFlag=True

                    signal_sequence = [i+1 if i+1<signal_num else -1 for i in self.sequence_id]
                    return signal_sequence

                else:
                    self.PassFlag = False
                    self.RedFlag = False
                    self.GreenFlag = False
                    signal_sequence=[]
                    for i in list(range(i, i+cosd_sign_num)):
                        if i<signal_num:
                            signal_sequence += [i]
                        else:
                            signal_sequence += [-1]
                    return signal_sequence





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
                signal_pos += [_clip+traj_length]
                traj_length += _clip
            return signal_pos

        def generate_signal_phase(self):
            # signal phase: [CycleTime(s), RedPhase(%), StartPhase(%)]
            signal_phase = []
            for i in range(signal_num):
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
            # StartPhase = list((200+np.array([-128.500000000000,-147.500000000000,-100.500000000000,-150.500000000000,-151.500000000000,-117.500000000000,-150.500000000000,-156.500000000000,-152.500000000000]))%100/100)
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
            self.acc_bound = self._Tq_to_Acc_bound()
            self.action = np.clip(action, *self.acc_bound)
            v_c = deepcopy(self.veh_state.v)
            self.veh_state.v += self.action*dt
            _vel= np.clip(self.veh_state.v, *[0,max_vel])
            if self.veh_state.v != _vel:
                self.NegVFlag = True
                # print("negative v=",self.veh_state.v, "action a=", self.action)
                self.action = (_vel-v_c)/dt
                self.veh_state.v = _vel
            else:
                # print("action a=", self.action)
                self.NegVFlag = False
            self.veh_state.dist_from_start += v_c*dt + 1/2*self.action*dt**2
            v_n = deepcopy(self.veh_state.v)
            self._E_consumption = self._get_E_consumption(self.action, v_c, v_n)
            if np.isnan(self.veh_state.dist_from_start) or np.isnan(self.veh_state.v):
                print("aaa")
            if self.veh_state.v<0:
                print("Too wierd")
            # Update plot @#$%


        def _Tq_to_Acc_bound(self):         # func(self.v)
            self._F_calculate()
            # _vel = np.clip(self.veh_state.v, *[0,max_vel])
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
            E_consumption_J = 0.5*EVeh_model.VOC_norm*(I_c+I_n)*dt
            if E_consumption_J<0 or np.isnan(E_consumption_J):
                print("aho")

            return E_consumption_J

class veh_state():
    def __init__(self):
        # self.v = 25 + 5 * np.random.rand()
        # self.v = max_vel * np.random.rand()
        # self.dist_from_start = 10 * np.random.rand()
        # self.v = max_vel*np.random.rand()
        self.v = 0
        self.dist_from_start = 0



def normalize_state(state):
    state = np.squeeze(state)
    if state.ndim==1:
        state = state[np.newaxis,:]
    state[:,0] = state[:,0]/max_vel
    # state[:,0] = state[:,0]/(max_vel/2)-1

    # self.state_new:
    if mod == "dim3":
        state[:,1] = state[:,1]/max_dist
        state[:,2] = state[:,2]/max_step/dt

    # self.state:
    elif mod == "dim5":
        # for i in range(cosd_sign_num):
        state[:,1+SIGNL_DIM*0] = 2*state[:,1+SIGNL_DIM*0]/max_interval
        state[:,2+SIGNL_DIM*0] = 2*state[:,2+SIGNL_DIM*0]/max_cycle
        state[:,3+SIGNL_DIM*0] = 2*state[:,3+SIGNL_DIM*0]/max_red

        state[:,1+SIGNL_DIM*1] = 2*state[:,1+SIGNL_DIM*1]/2000
        state[:,2+SIGNL_DIM*1] = 2*state[:,2+SIGNL_DIM*1]/max_cycle
        state[:,3+SIGNL_DIM*1] = 2*state[:,3+SIGNL_DIM*1]/max_red

        # state[:,1+SIGNL_DIM*0] = 2*state[:,1+SIGNL_DIM*0]/max_interval-1
        # state[:,2+SIGNL_DIM*0] = 2*state[:,2+SIGNL_DIM*0]/max_cycle-1
        # state[:,3+SIGNL_DIM*0] = 2*state[:,3+SIGNL_DIM*0]/max_red-1
        #
        # state[:,1+SIGNL_DIM*1] = 2*state[:,1+SIGNL_DIM*1]/2000-1
        # state[:,2+SIGNL_DIM*1] = 2*state[:,2+SIGNL_DIM*1]/max_cycle-1
        # state[:,3+SIGNL_DIM*1] = 2*state[:,3+SIGNL_DIM*1]/max_red-1


    _state = state
    return _state

