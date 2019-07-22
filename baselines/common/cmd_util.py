"""
Helpers for scripts like run_atari.py.
"""

import os
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import gym
from gym.wrappers import FlattenDictWrapper
from baselines import logger
from baselines.bench import Monitor
from baselines.common import set_global_seeds
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common import retro_wrappers
from baselines.common.wrappers import ClipActionsWrapper
import numpy as np
from scipy.interpolate import interp2d, interp1d


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


def make_vec_env(env_id, env_type, num_env, seed,
                 wrapper_kwargs=None,
                 env_kwargs=None,
                 start_index=0,
                 reward_scale=1.0,
                 flatten_dict_observations=True,
                 gamestate=None,
                 initializer=None,
                 force_dummy=False):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari and MuJoCo.
    """
    wrapper_kwargs = wrapper_kwargs or {}
    env_kwargs = env_kwargs or {}
    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    seed = seed + 10000 * mpi_rank if seed is not None else None
    logger_dir = logger.get_dir()
    def make_thunk(rank, initializer=None):
        return lambda: make_env(
            env_id=env_id,
            env_type=env_type,
            mpi_rank=mpi_rank,
            subrank=rank,
            seed=seed,
            reward_scale=reward_scale,
            gamestate=gamestate,
            flatten_dict_observations=flatten_dict_observations,
            wrapper_kwargs=wrapper_kwargs,
            env_kwargs=env_kwargs,
            logger_dir=logger_dir,
            initializer=initializer
        )

    set_global_seeds(seed)
    if not force_dummy and num_env > 1:
        return SubprocVecEnv([make_thunk(i + start_index, initializer=initializer) for i in range(num_env)])
    else:
        return DummyVecEnv([make_thunk(i + start_index, initializer=None) for i in range(num_env)])


def make_env(env_id, env_type, mpi_rank=0, subrank=0, seed=None, reward_scale=1.0, gamestate=None, flatten_dict_observations=True, wrapper_kwargs=None, env_kwargs=None, logger_dir=None, initializer=None):
    if initializer is not None:
        initializer(mpi_rank=mpi_rank, subrank=subrank)

    wrapper_kwargs = wrapper_kwargs or {}
    env_kwargs = env_kwargs or {}
    if ':' in env_id:
        import re
        import importlib
        module_name = re.sub(':.*','',env_id)
        env_id = re.sub('.*:', '', env_id)
        importlib.import_module(module_name)
    if env_type == 'atari':
        env = make_atari(env_id)
    elif env_type == 'retro':
        import retro
        gamestate = gamestate or retro.State.DEFAULT
        env = retro_wrappers.make_retro(game=env_id, max_episode_steps=10000, use_restricted_actions=retro.Actions.DISCRETE, state=gamestate)
    else:
        env = gym.make(env_id, **env_kwargs)

    if flatten_dict_observations and isinstance(env.observation_space, gym.spaces.Dict):
        keys = env.observation_space.spaces.keys()
        env = gym.wrappers.FlattenDictWrapper(env, dict_keys=list(keys))

    env.seed(seed + subrank if seed is not None else None)
    env = Monitor(env,
                  logger_dir and os.path.join(logger_dir, str(mpi_rank) + '.' + str(subrank)),
                  allow_early_resets=True)


    if env_type == 'atari':
        env = wrap_deepmind(env, **wrapper_kwargs)
    elif env_type == 'retro':
        if 'frame_stack' not in wrapper_kwargs:
            wrapper_kwargs['frame_stack'] = 1
        env = retro_wrappers.wrap_deepmind_retro(env, **wrapper_kwargs)

    if isinstance(env.action_space, gym.spaces.Box):
        env = ClipActionsWrapper(env)

    if reward_scale != 1:
        env = retro_wrappers.RewardScaler(env, reward_scale)

    return env


def make_mujoco_env(env_id, seed, reward_scale=1.0):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.
    """
    rank = MPI.COMM_WORLD.Get_rank()
    myseed = seed  + 1000 * rank if seed is not None else None
    set_global_seeds(myseed)
    env = gym.make(env_id)
    logger_path = None if logger.get_dir() is None else os.path.join(logger.get_dir(), str(rank))
    env = Monitor(env, logger_path, allow_early_resets=True)
    env.seed(seed)
    if reward_scale != 1.0:
        from baselines.common.retro_wrappers import RewardScaler
        env = RewardScaler(env, reward_scale)
    return env

def make_robotics_env(env_id, seed, rank=0):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.
    """
    set_global_seeds(seed)
    env = gym.make(env_id)
    env = FlattenDictWrapper(env, ['observation', 'desired_goal'])
    env = Monitor(
        env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
        info_keywords=('is_success',))
    env.seed(seed)
    return env

def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def atari_arg_parser():
    """
    Create an argparse.ArgumentParser for run_atari.py.
    """
    print('Obsolete - use common_arg_parser instead')
    return common_arg_parser()

def mujoco_arg_parser():
    print('Obsolete - use common_arg_parser instead')
    return common_arg_parser()

def common_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='Reacher-v2')
    parser.add_argument('--env_type', help='type of environment, used when the environment type cannot be automatically determined', type=str)
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--alg', help='Algorithm', type=str, default='ppo2')
    parser.add_argument('--num_timesteps', type=float, default=1e6),
    parser.add_argument('--network', help='network type (mlp, cnn, lstm, cnn_lstm, conv_only)', default=None)
    parser.add_argument('--gamestate', help='game state to load (so far only used in retro games)', default=None)
    parser.add_argument('--num_env', help='Number of environment copies being run in parallel. When not specified, set to number of cpus for Atari, and to 1 for Mujoco', default=None, type=int)
    parser.add_argument('--reward_scale', help='Reward scale factor. Default: 1.0', default=1.0, type=float)
    parser.add_argument('--save_path', help='Path to save trained model to', default=None, type=str)
    parser.add_argument('--save_video_interval', help='Save video every x steps (0 = disabled)', default=0, type=int)
    parser.add_argument('--save_video_length', help='Length of recorded video. Default: 200', default=200, type=int)
    parser.add_argument('--log_path', help='Directory to save learning curve data.', default=None, type=str)
    parser.add_argument('--play', default=False, action='store_true')

    # For Signalized-Eco:
    # parser.add_argument(
    #     "--SHOW",
    #     type = bool,
    #     default = True,
    #     help = "Plot Env, Training curves"
    # )
    #
    # parser.add_argument(
    #     "--plot-mode",
    #     type = str,
    #     default = "all",
    #     help = "The mode of plotting traffic lights"
    # )
    #
    #
    # parser.add_argument(
    #     "--debug",
    #     type = bool,
    #     default = True,
    #     help = "Plot Env, Training curves"
    # )
    #
    #
    #
    # # Environment argument
    # parser.add_argument(
    #     "--lane-num",
    #     type = int,
    #     default = 1,
    #     help = "The number of lanes"
    # )
    #
    # parser.add_argument(
    #     "--traj-length",
    #     type = float,
    #     default = 1000,
    #     help = "The length of trajectory"
    # )
    #
    # parser.add_argument(
    #     "--signal-num",
    #     type = int,
    #     default = 3,
    #     help = "The number of traffic signals"
    # )
    #
    # parser.add_argument(
    #     "--SIGNL-DIM",
    #     type=int,
    #     default=4,
    #     help="The dimension of a traffic signal represented as state: [dist_m, cycle_s, red_%, current_%]"
    # )
    #
    # parser.add_argument(
    #     "--STATE-DIM",
    #     type=int,
    #     default=4*2+1,
    #     help="The dimension of state"
    # )
    # parser.add_argument(
    #     "--ACTION-DIM",
    #     type=int,
    #     default=1,
    #     help="The dimension of action"
    # )
    #
    # # Interval:[100,500]
    # parser.add_argument(
    #     "--max_interval",
    #     type = float,
    #     default = 1000,
    #     help = "The max velofcity"
    # )
    # # Traffic cycle: [80,160]
    # parser.add_argument(
    #     "--max_cycle",
    #     type = float,
    #     default = 180,
    #     help = "The max velocity"
    # )
    # # Traffic red proportion: [0.3,0.7]
    # parser.add_argument(
    #     "--max_red",
    #     type = float,
    #     default = 1,
    #     help = "The max velocity"
    # )
    #
    # # Agent_EV
    # parser.add_argument(
    #     "--EVeh",
    #     type = EVeh_model,
    #     help = "Electric vehicle parameters"
    # )
    #
    # parser.add_argument(
    #     "--cosd-sign-num",
    #     type = int,
    #     default = 2,
    #     help = "The number of considered traffic signals"
    # )
    #
    # parser.add_argument(
    #     "--dt",
    #     type = float,
    #     default = 0.1,
    #     help = "The time step"
    # )
    #
    # parser.add_argument(
    #     "--max_vel",
    #     type = float,
    #     default = 12500*np.pi/30/EVeh_model.M2WRatio*EVeh_model.WheelRadius,
    #     help = "The max velocity - m/s"
    # )
    #
    #
    #
    # # Learning argument
    # parser.add_argument(
    #     "--use-cuda",
    #     type = bool,
    #     default = False,
    #     help = "Use GPU or CPU"
    # )
    # parser.add_argument(
    #     "--actor-lr",
    #     type = float,
    #     default = 1e-3,
    #     help = "The learning rate"
    # )
    #
    # parser.add_argument(
    #     "--critic-lr",
    #     type = float,
    #     default = 1e-2,
    #     help = "The learning rate"
    # )
    #
    # parser.add_argument(
    #     "--sample-len",
    #     type = int,
    #     default = 128,
    #     help = "The length of samples"
    # )
    #
    # parser.add_argument(
    #     "--GAMMA",
    #     type = float,
    #     default = 0.995,
    #     help = "The discounted reward parameter"
    # )
    #
    # # @#$%
    # parser.add_argument(
    #     "--target-tau",
    #     type = float,
    #     default = 1e-3,
    #     help = "The target update parameter"
    # )
    #
    # parser.add_argument(
    #     "--target-update-steps",
    #     type = float,
    #     default = 3.2,
    #     help = "The target update parameter"
    # )
    #
    # parser.add_argument(
    #     "--clip-param",
    #     type = float,
    #     default = 0.2,
    #     help = "The clip param for PPO"
    # )
    #
    #
    #
    # parser.add_argument(
    #     "--E-reward",
    #     type = float,
    #     default = 1e-4,
    #     help = "The reward ratio of E_consumption"
    # )
    # # parser.add_argument(
    # #     "--Aux-reward",
    # #     type = float,
    # #     default = 5e-3,
    # #     help = "The reward ratio of Aux_consumption"
    # # )
    # parser.add_argument(
    #     "--T-reward",
    #     type = float,
    #     default = 1,
    #     help = "The reward ratio of Time"
    # )
    #
    # parser.add_argument(
    #     "--Red-reward",
    #     type = float,
    #     default = 50,
    #     help = "The reward ratio of Time"
    # )
    #
    # parser.add_argument(
    #     "--Pass-reward",
    #     type = float,
    #     default = 10,
    #     help = "The reward ratio of Time"
    # )
    #
    # parser.add_argument(
    #     "--End-reward",
    #     type = float,
    #     default = 50,
    #     help = "The reward ratio of End"
    # )
    #
    # parser.add_argument(
    #     "--memory-capacity",
    #     type=int,
    #     default=2000,
    #     help="The capacity of memory"
    # )
    #
    # parser.add_argument(
    #     "--batch-size",
    #     type=int,
    #     default=32,
    #     help="The capacity of memory"
    # )
    #
    # parser.add_argument(
    #     "--evaluate-eps",
    #     type=int,
    #     default=5,
    #     help="The episodes after to evaluate"
    # )

    return parser

def robotics_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='FetchReach-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    return parser


def parse_unknown_args(args):
    """
    Parse arguments not consumed by arg parser into a dictionary
    """
    retval = {}
    preceded_by_key = False
    for arg in args:
        if arg.startswith('--'):
            if '=' in arg:
                key = arg.split('=')[0][2:]
                value = arg.split('=')[1]
                retval[key] = value
            else:
                key = arg[2:]
                preceded_by_key = True
        elif preceded_by_key:
            retval[key] = arg
            preceded_by_key = False

    return retval
