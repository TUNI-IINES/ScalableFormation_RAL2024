import pickle
import matplotlib.pyplot as plt

from scenarios_unicycle.CCTA2024_Controller import FeedbackInformation, SceneSetup, np
from scenarios_unicycle.CCTA2024_FormationObstacleLidar_scenario import SimSetup, ExpSetup, import_scenario
from matplotlib.gridspec import GridSpec
from nebolab_experiment_setup import NebolabSetup
from simulator.plot_2D_unicycle import draw2DUnicyle
import imutils
import cv2
import colorsys


class ReAnimate:
    red = [(0, 70, 50), (10, 255, 255)]
    blue = [(100, 150, 0), (140, 255, 255)]
    green = [(36, 25, 25), (70, 255, 255)]
    yellow = [(20, 100, 100), (30, 255, 255)]

    def __init__(self):
        self.__cur_time = 0.
        self.feedback_information = FeedbackInformation()

        # Initiate ranging sensors for the obstacles
        # self.__rangesens = DetectObstacle(detect_max_dist=SceneSetup.default_range,
        #                                   angle_res_rad=2 * np.pi / SceneSetup.sensor_resolution)
        # for i in range(len(SceneSetup.static_obstacles)):
        #     self.__rangesens.register_obstacle_bounded('obs' + str(i), SceneSetup.static_obstacles[i])

        # Display sensing data
        self.__pl_sens = dict()

        # Initiate the plotting
        self.__initiate_plot()

        # flag to check if simulation is still running
        self.is_running = True

    def __initiate_plot(self):
        # For now plot 2D with 2x2 grid space, to allow additional plot later on
        # rowNum, colNum = 2, 3
        rowNum, colNum = 6, 2
        self.fig = plt.figure(figsize=(colNum * 4, rowNum), dpi=100)
        gs = GridSpec(rowNum, colNum, figure=self.fig)

        # MAIN 2D PLOT FOR UNICYCLE ROBOTS
        # ------------------------------------------------------------------------------------
        # ax_2D = self.fig.add_subplot(gs[0:2, 0:2])  # Always on
        ax_2D = self.fig.add_subplot(gs[0:3, :])  # Always on
        # Only show past several seconds trajectory
        # trajTail_datanum = int(SimSetup.trajectory_trail_lenTime / SimSetup.Ts)
        trajTail_datanum = int(SimSetup.trajectory_trail_lenTime / SimSetup.Ts) * 10
        #
        self.__drawn_2D = draw2DUnicyle(ax_2D, SceneSetup.init_pos, SceneSetup.init_theta,
                                        field_x=NebolabSetup.FIELD_X, field_y=NebolabSetup.FIELD_Y,
                                        pos_trail_nums=trajTail_datanum)
        # self.__drawn_ellipse_form = {}
        # for i in range(SceneSetup.form_num):
        #     self.__drawn_ellipse_form[i] = drawMovingEllipse( ax_2D, np.zeros(3), SceneSetup.major_l[i], SceneSetup.minor_l[i], 0.)

        # Draw goals and obstacles
        self.__pl_goal = {}
        for i in range(SceneSetup.robot_num):
            # ax_2D.add_patch(plt.Circle((SceneSetup.goal_pos[i][0], SceneSetup.goal_pos[i][1]), 0.03, color='g'))
            self.__pl_goal[i], = ax_2D.plot(SceneSetup.goal_pos[i, 0], SceneSetup.goal_pos[i, 1], 'go')
        for obs in SceneSetup.static_obstacles:
            ax_2D.plot(obs[:, 0], obs[:, 1], 'k')

        # Display simulation time
        self.__drawn_time = ax_2D.text(0.78, 0.99, 't = 0 s', color='k', fontsize='large',
                                       horizontalalignment='left', verticalalignment='top', transform=ax_2D.transAxes)

        # Draw communication lines within formations
        self.__drawn_comm_lines = {}
        for i in range(SceneSetup.robot_num):
            for j in range(SceneSetup.robot_num):
                if (i < j) and (SceneSetup.form_A[i, j] > 0):
                    self.__drawn_comm_lines[str(i) + '_' + str(j)], = ax_2D.plot([-i, -i], [j, j],
                                                                                 color='k', linewidth=0.5 /
                                                                                                      SceneSetup.max_form_epsilon[
                                                                                                          i][j])

        # Display sensing data
        self.__pl_sens = dict()
        __colorList = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i in range(SceneSetup.robot_num):
            self.__pl_sens[i], = ax_2D.plot(0, 0, '.', color=__colorList[i], markersize=0.25)

        # ADDITIONAL PLOT
        # ------------------------------------------------------------------------------------
        # Plot the distance between robots
        # self.__ax_dist = self.fig.add_subplot(gs[0, 2])
        self.__ax_dist = self.fig.add_subplot(gs[3:6, 0])
        self.__ax_dist.set(xlabel="t [s]", ylabel="distance [m]")
        colorList = plt.rcParams['axes.prop_cycle'].by_key()['color']

        self.__drawn_distance_lines = {}
        cnt = 0
        for i in range(SceneSetup.robot_num):
            for j in range(SceneSetup.robot_num):
                if (i < j) and (SceneSetup.form_A[i, j] > 0):
                    self.__drawn_distance_lines[str(i) + '_' + str(j)], = self.__ax_dist.plot(0, 0,
                                                                                              color=colorList[cnt],
                                                                                              label='$i={},\ j={}$'.format(
                                                                                                  i + 1, j + 1))
                    cnt += 1
        # Draw the specified band
        array_req_dist = np.unique(SceneSetup.form_A)
        array_req_dist = np.delete(array_req_dist, 0)
        array_max_eps = np.unique(SceneSetup.max_form_epsilon)
        array_max_eps = np.delete(array_max_eps, 0)
        self.__prev_fill = list()

        # for idx, dist in enumerate(array_req_dist):
        #     if idx == 0:  # only put 1 label
        #         self.__ax_dist.fill_between([0, SimSetup.tmax], [dist - array_max_eps[idx]] * 2,
        #                                     [dist + array_max_eps[idx]] * 2,
        #                                     alpha=0.12, color='k', linewidth=0, label='specified distance')
        #     else:
        #         self.__ax_dist.fill_between([0, SimSetup.tmax], [dist - array_max_eps[idx]] * 2,
        #                                     [dist + array_max_eps[idx]] * 2,
        #                                     alpha=0.12, color='k', linewidth=0)
        # set y-axis
        self.__ax_dist.set(ylim=(max(min(array_req_dist) - max(array_max_eps) - 0.1, 0.0),
                                 max(array_req_dist) + max(array_max_eps) + 0.1))
        self.__ax_dist.grid(True)
        self.__ax_dist.legend(loc=(0.65, 0.18), prop={'size': 6})

        # Plot the h_function for obstacles
        self.__ax_hobs = self.fig.add_subplot(gs[3:6, 1])
        self.__ax_hobs.set(xlabel="t [s]", ylabel="h_obs")
        l_style, cnt = ['-', ':', '.'], 0
        self.__drawn_h_obs_lines = dict()

        for i in range(SceneSetup.robot_num):
            self.__drawn_h_obs_lines[str(i)], = self.__ax_hobs.plot(0, 0, '-', color=colorList[cnt],
                                                                    label='robot ${}$'.format(i + 1))
            cnt += 1
        # set grid and legend
        self.__ax_hobs.grid(True)
        self.__ax_hobs.legend(loc='upper left', prop={'size': 6})

        plt.tight_layout()

    def __update_plot(self, feedback):
        # UPDATE 2D Plotting: Formation and Robots
        # for i in range(SceneSetup.form_num):
        #    el_pos, el_th = feedback.get_form_i_state(i)
        #    self.__drawn_ellipse_form[i].update( el_pos, el_th )
        self.__drawn_2D.update(feedback.get_all_robot_pos(), feedback.get_all_robot_theta())
        self.__drawn_time.set_text('t = ' + f"{self.__cur_time:.1f}" + ' s')

        # update display of sensing data
        for i in range(SceneSetup.robot_num):
            sensed_pos = feedback.get_robot_i_detected_pos(i)
            self.__pl_sens[i].set_data(sensed_pos[:, 0], sensed_pos[:, 1])

        # Update communication lines within formations
        la_pos = feedback.get_all_lahead_pos()
        for i in range(SceneSetup.robot_num):
            for j in range(SceneSetup.robot_num):
                if (i < j) and (SceneSetup.form_A[i, j] > 0):
                    self.__drawn_comm_lines[str(i) + '_' + str(j)].set_data(
                        [la_pos[i][0], la_pos[j][0]], [la_pos[i][1], la_pos[j][1]])

        # get data from Log
        log_data, max_idx = self.log.get_all_data()

        # Update goal position (changes due to waypoints)
        if SceneSetup.USE_WAYPOINTS:
            for i in range(SceneSetup.robot_num):
                goal_i_x = log_data['goal_x_' + str(i)][max_idx - 1]
                goal_i_y = log_data['goal_y_' + str(i)][max_idx - 1]
                # print(i, goal_i_x, goal_i_y)
                self.__pl_goal[i].set_data(goal_i_x, goal_i_y)

        # Setup for moving window horizon
        if self.__cur_time < SimSetup.timeseries_window:
            t_range = (-0.1, SimSetup.timeseries_window + 0.1)
            min_idx = 0
        else:
            t_range = (self.__cur_time - (SimSetup.timeseries_window + 0.1), self.__cur_time + 0.1)
            min_idx = max_idx - round(SimSetup.timeseries_window / SimSetup.Ts)

        # Update the distance between robots
        fill_segment = list()
        for i in range(SceneSetup.robot_num):
            for j in range(SceneSetup.robot_num):
                if (i < j) and (SceneSetup.form_A[i, j] > 0):
                    dist = [np.sqrt(
                        (log_data['pos_x_' + str(i)][k] - log_data['pos_x_' + str(j)][k]) ** 2 +
                        (log_data['pos_y_' + str(i)][k] - log_data['pos_y_' + str(j)][k]) ** 2)
                        for k in range(min_idx, max_idx)]

                    self.__drawn_distance_lines[str(i) + '_' + str(j)].set_data(
                        log_data['time'][min_idx:max_idx], dist)

                    if SimSetup.eps_visualization:  # TODO WIDHI: check notes in SimSetup
                        visible_length = int(SimSetup.timeseries_window // SimSetup.Ts)
                        eps_hist = feedback.get_robot_i_j_eps_hist(i, j)[-visible_length:] + \
                                   feedback.get_robot_i_j_eps_hist(j, i)[-visible_length:]

                        fill_instance = self.__ax_dist.fill_between(
                            np.linspace(max(0, self.__cur_time - SimSetup.timeseries_window), self.__cur_time,
                                        num=eps_hist.size),
                            (SceneSetup.form_A[i, j] - eps_hist), (SceneSetup.form_A[i, j] + eps_hist),
                            alpha=0.1, color='k', linewidth=0)
                        fill_segment.append(fill_instance)

        for fill_instance in self.__prev_fill:
            fill_instance.remove()
        self.__prev_fill = fill_segment

        # Update the h functions
        max_h_val = 0.

        for i in range(SceneSetup.robot_num):
            h_val = log_data["h_staticobs_" + str(i)][min_idx:max_idx]
            max_h_val = max(max_h_val, max(h_val))
            self.__drawn_h_obs_lines[str(i)].set_data(log_data['time'][min_idx:max_idx], h_val)

        # Move the time-series window
        self.__ax_dist.set(xlim=t_range)
        self.__ax_hobs.set(xlim=t_range, ylim=(-0.1, max_h_val + 0.1))


def preamble_setting(filename):  # when manual plotting is needed
    """

    :return:
    """
    # List of scenario mode
    # SceneSetup.SCENARIO_MODE = 0  # basic leader-following no attack
    # SceneSetup.SCENARIO_MODE = 1 # with attack but no defense
    # SceneSetup.SCENARIO_MODE = 2 # with attack and with defense
    # SceneSetup.SCENARIO_MODE = 3 # with attack and another version of defense

    # SimSetup.sim_defname = 'animation_result/Resilient_scenario/sim_' + str(SceneSetup.SCENARIO_MODE)
    # SimSetup.sim_fdata_log = SimSetup.sim_defname + '_vis.pkl'

    # Temporary fix for experiment data TODO: fix this later
    import_scenario(directory='scenarios_unicycle/saved_pkl/', filename=filename)
    SimSetup.sim_defname = 'scenarios_unicycle/saved_pkl/' + filename  # + '_beta2'
    SimSetup.sim_fdata_log = SimSetup.sim_defname + '.pkl'
    SimSetup.sim_fdata_vis = SimSetup.sim_defname + '.pkl'
    SimSetup.sim_fname_output = r'' + SimSetup.sim_defname + '.gif'


def plot_pickle_log_time_series_batch_keys(ax, datalog_data, __end_idx, pre_string):
    """

    :param ax:
    :param datalog_data:
    :param __end_idx:
    :param pre_string:
    :return:
    """
    # check all matching keystring
    time_data = datalog_data['time'][:__end_idx]
    matches = [key for key in datalog_data if key.startswith(pre_string)]
    data_min, data_max = 0., 0.
    for key in matches:
        key_data = datalog_data[key][:__end_idx]
        # key_data = key_data.reshape(key_data.shape[0], -1) if len(key_data.shape) > 2 else key_data
        ax.plot(time_data, key_data, label=key.strip(pre_string) if len(key.strip(pre_string)) > 0 else pre_string)
        # update min max for plotting
        data_min = min(data_min, min(i for i in key_data if i is not None))
        data_max = max(data_max, max(i for i in key_data if i is not None))
        # adjust time window
    ax.grid(True)
    ax.set(xlim=(time_data[0] - 0.1, time_data[-1] + 0.1),
           ylim=(data_min - 0.1, data_max + 0.1))


def plot_pickle_log_time_series_batch_robotid(ax, datalog_data, __end_idx, pre_string, id_name=None):
    """

    :param ax:
    :param datalog_data:
    :param __end_idx:
    :param pre_string:
    :param id_name:
    :return:
    """
    # check all matching keystring
    time_data = datalog_data['time'][:__end_idx]
    data_min, data_max = 0., 0.
    if id_name is None: id_name = [str(i) for i in range(SceneSetup.robot_num)]
    for i in range(SceneSetup.robot_num):
        key = pre_string + str(i)
        key_data = datalog_data[key][:__end_idx]
        ax.plot(time_data, key_data, color=SceneSetup.robot_color[i], label=id_name[i])
        # update min max for plotting
        data_min = min(data_min, min(i for i in key_data if i is not None))
        data_max = max(data_max, max(i for i in key_data if i is not None))
        # adjust time window
    ax.grid(True)
    ax.set(xlim=(time_data[0] - 0.1, time_data[-1] + 0.1),
           ylim=(data_min - 0.1, data_max + 0.1))


def plot_pickle_robot_distance(ax, datalog_data, __end_idx, pre_pos_x, pre_pos_y):
    """

    :param ax:
    :param datalog_data:
    :param __end_idx:
    :param pre_pos_x:
    :param pre_pos_y:
    :return:
    """
    dist_max = 0.
    time_data = datalog_data['time'][:__end_idx]
    for i in range(SceneSetup.robot_num):
        for j in range(SceneSetup.robot_num):
            if i < j:
                dist = [np.sqrt(
                    (datalog_data[pre_pos_x + str(i)][k] - datalog_data[pre_pos_x + str(j)][k]) ** 2 +
                    (datalog_data[pre_pos_y + str(i)][k] - datalog_data[pre_pos_y + str(j)][k]) ** 2)
                    for k in range(__end_idx)]
                dist_max = max(dist_max, max(dist))
                # update line plot
                ax.plot(time_data, dist, label='$i={},\ j={}$'.format(i + 1, j + 1))
    # update axis length
    ax.grid(True)
    ax.set(xlim=(time_data[0] - 0.1, time_data[-1] + 0.1),
           ylim=(-0.1, dist_max + 0.1))


def scenario_pkl_plot():
    """
    Plot the logged data
    """
    # ---------------------------------------------------
    with open(SimSetup.sim_fdata_vis, 'rb') as f:
        visData = pickle.load(f)
    __stored_data = visData['stored_data']
    # __end_idx = visData['last_idx']
    __end_idx = len([t for t in __stored_data["time"] if t is not None])

    # set __end_idx manually
    t_stop = SimSetup.tmax
    __end_idx = t_stop * ExpSetup.ROS_RATE - 1
    # __end_idx = t_stop * ExpSetup.LiDAR_RATE - 1
    # print(SceneSetup.robot_color)

    print(len(__stored_data['time']))
    # Print all key datas
    print(len([t for t in __stored_data["time"] if t is not None]), __end_idx)
    print(
        f'The file {SimSetup.sim_fdata_vis} contains the following logs for {__stored_data["time"][__end_idx]:.2f} s:')
    print(__stored_data.keys())
    # ---------------------------------------------------
    figure_short = (6.4, 3.4)
    figure_size = (6.4, 4.8)
    FS = 16  # font size
    LW = 1.5  # line width
    leg_size = 8

    bias_robot_name = ['red robot', 'blue robot (biased)', 'green robot (biased)']
    robot_name = ['red robot', 'blue robot', 'green robot']

    # PLOT THE POSITION
    # ---------------------------------------------------¨
    num_goal = SceneSetup.goal_pos.shape[0]
    fig, ax = plt.subplots(2, figsize=figure_short)
    plt.rcParams.update({'font.size': FS})
    # plt.rcParams['text.usetex'] = True
    # plot
    plot_pickle_log_time_series_batch_robotid(ax[0], __stored_data, __end_idx, 'pos_x_')
    plot_pickle_log_time_series_batch_robotid(ax[1], __stored_data, __end_idx, 'pos_y_')
    # plot goal pos
    for goal_id in range(SceneSetup.goal_pos.shape[0]):
        ax[0].plot([0, t_stop], [SceneSetup.goal_pos[goal_id, 0], SceneSetup.goal_pos[goal_id, 0]], ':', color='k',
                   label='goal position')
        ax[1].plot([0, t_stop], [SceneSetup.goal_pos[goal_id, 1], SceneSetup.goal_pos[goal_id, 1]], ':', color='k',
                   label='goal position')
    # label
    ax[0].set(ylabel='X-position [m]')
    ax[1].set(xlabel="t [s]", ylabel='Y-position [m]')
    ax[0].legend(loc='best', prop={'size': leg_size})
    # plt.show()
    figname = SimSetup.sim_defname + '_pos.pdf'
    pngname = SimSetup.sim_defname + '_pos.png'
    # plt.savefig(figname, bbox_inches="tight", dpi=300)
    plt.savefig(pngname, bbox_inches="tight", dpi=300)
    print('export figure: ' + pngname, flush=True)

    # PLOT THE VELOCITY
    # ---------------------------------------------------
    fig, ax = plt.subplots(2, figsize=figure_short)
    plt.rcParams.update({'font.size': FS})
    # plt.rcParams['text.usetex'] = True
    # plot
    plot_pickle_log_time_series_batch_robotid(ax[0], __stored_data, __end_idx, 'u_nom_x_')
    plot_pickle_log_time_series_batch_robotid(ax[1], __stored_data, __end_idx, 'u_nom_y_')
    # label
    ax[0].set(ylabel='u_x [m/s]')
    ax[1].set(xlabel="t [s]", ylabel='u_y [m/s]')
    ax[1].legend(loc='best', prop={'size': leg_size})
    # plt.show()
    # figname = SimSetup.sim_defname + '_u_nom.pdf'
    pngname = SimSetup.sim_defname + '_u_nom.png'
    plt.savefig(pngname, bbox_inches="tight", dpi=300)
    print('export figure: ' + pngname, flush=True)

    # # PLOT THE DETECTION
    # # ---------------------------------------------------
    fig = plt.figure(figsize=figure_short)
    plt.rcParams.update({'font.size': FS})
    # plt.rcParams['text.usetex'] = True
    ax = plt.gca()
    # plot
    plot_pickle_log_time_series_batch_keys(ax, __stored_data, __end_idx, 'h_staticobs_')
    ax.set(xlabel="t [s]", ylabel='h LiDAR')
    ax.legend(loc='best', prop={'size': leg_size})
    # plt.show()
    pngname = SimSetup.sim_defname + '_staticobs.png'
    plt.savefig(pngname, bbox_inches="tight", dpi=300)
    print('export figure: ' + pngname, flush=True)

    # # PLOT THE EPS
    # # ---------------------------------------------------
    if SceneSetup.USECBF_FORMATION:
        fig = plt.figure(figsize=figure_short)
        plt.rcParams.update({'font.size': FS})
        # plt.rcParams['text.usetex'] = True
        ax = plt.gca()
        # plot
        plot_pickle_log_time_series_batch_keys(ax, __stored_data, __end_idx, 'eps_')
        ax.set(xlabel="t [s]", ylabel='distance [m]')
        ax.legend(loc='best', prop={'size': leg_size})
        # plt.show()
        pngname = SimSetup.sim_defname + '_eps.png'
        plt.savefig(pngname, bbox_inches="tight", dpi=300)
        print('export figure: ' + pngname, flush=True)

        # # PLOT THE H EPS CEIL
        # # ---------------------------------------------------
        fig = plt.figure(figsize=figure_short)
        plt.rcParams.update({'font.size': FS})
        # plt.rcParams['text.usetex'] = True
        ax = plt.gca()
        # plot
        plot_pickle_log_time_series_batch_keys(ax, __stored_data, __end_idx, 'h_eps_ceil_')
        ax.set(xlabel="t [s]", ylabel="h_eps_ceil")
        ax.legend(loc='best', prop={'size': leg_size})
        # plt.show()
        pngname = SimSetup.sim_defname + '_h_eps_ceil.png'
        plt.savefig(pngname, bbox_inches="tight", dpi=300)
        print('export figure: ' + pngname, flush=True)

        # # PLOT THE H EPS FLOOR
        # # ---------------------------------------------------
        fig = plt.figure(figsize=figure_short)
        plt.rcParams.update({'font.size': FS})
        # plt.rcParams['text.usetex'] = True
        ax = plt.gca()
        # plot
        plot_pickle_log_time_series_batch_keys(ax, __stored_data, __end_idx, 'h_eps_floor_')
        ax.set(xlabel="t [s]", ylabel="h_eps_floor")
        ax.legend(loc='best', prop={'size': leg_size})
        # plt.show()
        pngname = SimSetup.sim_defname + '_h_eps_floor.png'
        plt.savefig(pngname, bbox_inches="tight", dpi=300)
        print('export figure: ' + pngname, flush=True)

        # # PLOT THE H FMU
        # # ---------------------------------------------------
        fig = plt.figure(figsize=figure_short)
        plt.rcParams.update({'font.size': FS})
        # plt.rcParams['text.usetex'] = True
        ax = plt.gca()
        # plot
        plot_pickle_log_time_series_batch_keys(ax, __stored_data, __end_idx, 'h_fmu_')
        ax.set(xlabel="t [s]", ylabel="h_fmu")
        ax.legend(loc='best', prop={'size': leg_size})
        # plt.show()
        pngname = SimSetup.sim_defname + '_h_fmu.png'
        plt.savefig(pngname, bbox_inches="tight", dpi=300)
        print('export figure: ' + pngname, flush=True)

        # # PLOT THE H FML
        # # ---------------------------------------------------
        fig = plt.figure(figsize=figure_short)
        plt.rcParams.update({'font.size': FS})
        # plt.rcParams['text.usetex'] = True
        ax = plt.gca()
        # plot
        plot_pickle_log_time_series_batch_keys(ax, __stored_data, __end_idx, 'h_fml_')
        ax.set(xlabel="t [s]", ylabel="h_fml")
        ax.legend(loc='best', prop={'size': leg_size})
        # plt.show()
        pngname = SimSetup.sim_defname + '_h_fml.png'
        plt.savefig(pngname, bbox_inches="tight", dpi=300)
        print('export figure: ' + pngname, flush=True)

    # # PLOT THE H STATICOBS
    # # ---------------------------------------------------
    if SceneSetup.USECBF_STATICOBS:
        fig = plt.figure(figsize=figure_short)
        plt.rcParams.update({'font.size': FS})
        # plt.rcParams['text.usetex'] = True
        ax = plt.gca()
        # plot
        plot_pickle_log_time_series_batch_keys(ax, __stored_data, __end_idx, 'h_staticobs_')
        ax.set(xlabel="t [s]", ylabel="h_staticobs")
        ax.legend(loc='best', prop={'size': leg_size})
        # plt.show()
        pngname = SimSetup.sim_defname + '_h_staticobs.png'
        plt.savefig(pngname, bbox_inches="tight", dpi=300)
        print('export figure: ' + pngname, flush=True)

    # PLOT THE DISTANCE
    # ---------------------------------------------------

    fig = plt.figure(figsize=figure_short)
    plt.rcParams.update({'font.size': FS})
    # plt.rcParams['text.usetex'] = True
    ax = plt.gca()
    # plot
    plot_pickle_log_time_series_batch_keys(ax, __stored_data, __end_idx, 'lidar_')
    ax.set(xlabel="t [s]", ylabel="LiDAR")
    ax.legend(loc='best', prop={'size': leg_size})
    # plt.show()
    pngname = SimSetup.sim_defname + '_lidar.png'
    plt.savefig(pngname, bbox_inches="tight", dpi=300)
    print('export figure: ' + pngname, flush=True)

    fig = plt.figure(figsize=figure_short)
    plt.rcParams.update({'font.size': FS})
    # plt.rcParams['text.usetex'] = True
    ax = plt.gca()
    # plot
    plot_pickle_robot_distance(ax, __stored_data, __end_idx, 'pos_x_', 'pos_y_')
    ax.set(xlabel="t [s]", ylabel='distance [m]')
    ax.legend(loc='best', prop={'size': leg_size})
    # plt.show()
    pngname = SimSetup.sim_defname + '_dist.png'
    plt.savefig(pngname, bbox_inches="tight", dpi=300)
    print('export figure: ' + pngname, flush=True)

    plt.close('all')


def prep(input_frame, lower, upper):
    # All shapes
    # Lower = (0, 86, 6)
    # Upper = (150, 255, 255)

    # Original shape
    # Lower = (29, 86, 6)
    # Upper = (64, 255, 255)

    # (r, g, b) = (27, 69, 103)
    # 0 255 102
    # (r, g, b) = (255, 0, 0)
    # 0 255 255

    # # normalize
    # (r, g, b) = (r / 255, g / 255, b / 255)
    # # convert to hsv
    # (h, s, v) = colorsys.rgb_to_hsv(r, g, b)
    # # expand HSV range
    # (h, s, v) = (int(h * 179), int(s * 255), int(v * 255))
    # print('HSV : ', h, s, v)

    # output_frame = imutils.resize(input_frame, width=800)
    # output_frame = input_frame.copy()
    ## Todo 3.1.3 gaussian blur
    # blurred = cv2.blur(input_frame, (12, 12))
    # hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    hsv = cv2.cvtColor(input_frame, cv2.COLOR_BGR2HSV)
    output_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    h, w = output_mask.shape
    output_mask[int(0.2 * h):int(0.285 * h), int(0.39 * w):int(0.59 * w)] = 0
    output_mask[int(0.62 * h):int(0.7 * h), int(0.39 * w):int(0.59 * w)] = 0
    return output_mask


def find_cnts(input_mask):
    # input: src_img, contour_mode, approx_method
    # output: contours and hierarchy
    # SIMPLE: two endpoints of the line
    # NONE: all boundary points
    threshold = 100
    # canny_output = cv2.Canny(input_mask, threshold, threshold * 2)
    # cnts = cv2.findContours(canny_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cv2.findContours(input_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)
    return cnts


def contour(input_frame, input_mask, color):
    cnts = find_cnts(input_mask)

    # c = cnts[5]
    # c = max(cnts, key=cv2.contourArea)
    # frame1 = input_frame.copy()
    centers = []

    cnts = list(cnts)

    overlap = True
    while overlap:
        overlap = False
        __contours = []
        # print(type(_contours[0][0]))
        for c in cnts:
            __contours.append([np.average(c[idx:min(idx + 5, len(c))], axis=0)
                               for idx in range(0, len(c), 5)])
        # print(len(__contours), len(_contours))
        for idx, c1 in enumerate(__contours):
            if not overlap:
                for ix, c2 in enumerate(__contours):
                    if ix <= idx: continue
                    if not overlap:
                        for i1, pt1 in enumerate(c1):
                            if not overlap:
                                for i2, pt2 in enumerate(c2):
                                    if not overlap:
                                        if (pt1[0][0] - pt2[0][0]) ** 2 + (pt1[0][1] - pt2[0][1]) ** 2 < 10:
                                            c1_ = [p[0] for p in cnts[ix]]
                                            c2_ = [p[0] for p in cnts[idx]]
                                            ctr = np.array(c1_[i1:] + c2_ + c1_[:i1]).reshape(
                                                (-1, 1, 2)).astype(np.int32)
                                            del cnts[ix]
                                            del cnts[idx]
                                            cnts.append(ctr)
                                            overlap = True
                                            break


        # bboxes = []
        # for c in cnts:
        #     contours_poly = cv2.approxPolyDP(c, 3, True)
        #     bboxes.append(cv2.boundingRect(contours_poly))
        #
        # for idx, box in enumerate(bboxes):
        #     if not overlap:
        #         for c in cnts[idx + 1:]:
        #             if not overlap:
        #                 for pt in c:
        #                     if box[0] < pt[0][0] < box[0] + box[2] and box[1] < pt[0][1] < box[1] + box[3]:
        #                         ctr = np.array([p[0] for p in c] + [p[0] for p in cnts[idx]]).reshape((-1, 1, 2)).astype(np.int32)
        #                         del cnts[idx:idx+2]
        #                         cnts.append(ctr)
        #                         overlap = True
        #                         break

    cnts = tuple(cnts)

    for c in cnts:
        if cv2.contourArea(c) < 50 or cv2.contourArea(c) > 10000: continue

        ((x, y), radius) = cv2.minEnclosingCircle(c)  ## A different contour?
        # print(cnts)

        # Find center of contour using moments in opencvq
        M = cv2.moments(c)
        try:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            centers.append(center)
        except ZeroDivisionError as e:
            print("ZeroDivisionError")
            continue

        # # circle
        # cv2.circle(input_frame, (int(x), int(y)), int(radius), (0, 0, 255), 2)
        # cv2.circle(input_frame, center, 5, (0, 0, 255), -1)
        #
        # # draw rectangle with green line
        # contours_poly = cv2.approxPolyDP(c, 3, True)
        # boundRect = cv2.boundingRect(contours_poly)
        # cv2.rectangle(input_frame, (int(boundRect[0]), int(boundRect[1])),
        #               (int(boundRect[0] + boundRect[2]), int(boundRect[1] + boundRect[3])), (0, 255, 0), 1)
        #
        # # draw rotate rectangle with blue line
        # rect = cv2.minAreaRect(c)
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        # cv2.drawContours(input_frame, [box], 0, (255, 0, 0), 1)
        #
        # # draw Hull with pink line
        hull = cv2.convexHull(c)
        cv2.drawContours(input_frame, [hull], 0, color, 1)

    if len(centers) > 0:
        avg_center = (int(sum([c[0] for c in centers]) / len(centers)),
                      int(sum([c[1] for c in centers]) / len(centers)))
        return avg_center
    else:
        return None


def exp_video_pkl_plot(snap=False, beautify=False):
    """

    :return:
    """
    # videoloc = None

    import matplotlib.animation as animation

    if not beautify: videoloc = SimSetup.sim_defname + '_fixed.avi'  # TODO: fix the variable loc
    else: videoloc = SimSetup.sim_defname + '.mp4'  # TODO: fix the variable loc
    outname = SimSetup.sim_defname + 'snap_'
    # if SceneSetup.SCENARIO_MODE < 2:
    #     time_snap = [20]  # in seconds
    # else:
    #     time_snap = [SimSetup.tmax]  # in seconds
    time_snap = [10, 60, 140, 240, 330, 413]  # in seconds
    # goal_snap = [160, 160, 285, 285, 415, 415]  # in seconds
    goal_snap = [0, 0, 0, 2, 4, 4]  # in order
    past_t = [0, 50, 80, 100, 90, 83]  # in seconds
    frame_shift = 100  # accomodate on-sync video and data, video ALWAYS earlier
    data_freq = ExpSetup.ROS_RATE
    # data_freq = ExpSetup.LiDAR_RATE

    if videoloc is not None:
        import cv2
        from nebolab_experiment_setup import NebolabSetup

        # Initialize VIDEO
        cam = cv2.VideoCapture(videoloc)
        frame_per_second = cam.get(cv2.CAP_PROP_FPS)
        length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))

        current_step = -frame_shift

        # Initialize Pickle
        with open(SimSetup.sim_fdata_log, 'rb') as f:
            visData = pickle.load(f)
            print(visData['stored_data'].keys())
        __stored_data = visData['stored_data']
        __end_idx = visData['last_idx']
        print('Frames:', length, ", Time:", visData['last_idx'])
        # print('Keys', __stored_data.keys())
        SceneSetup.robot_color = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e']

        # Convert robot position log into pixel
        pos_pxl = {}
        for i in range(SceneSetup.robot_num):
            px_key, py_key = 'pos_x_' + str(i), 'pos_y_' + str(i)
            pos_pxl[px_key] = np.zeros(__end_idx)
            pos_pxl[py_key] = np.zeros(__end_idx)
            for j in range(__end_idx):
                pos_pxl[px_key][j], pos_pxl[py_key][j] = NebolabSetup.pos_m2pxl(__stored_data[px_key][j],
                                                                                __stored_data[py_key][j])
        if snap:
            # Proceed with video reading
            for idx, snap_point in enumerate(time_snap):
                # roll the video towards snap_point
                while current_step <= snap_point * frame_per_second:
                    ret, frame = cam.read()
                    if ret:
                        current_step += 1
                    else:
                        break

                # plot on snap_point
                fig = plt.figure()
                ax = plt.gca()

                b, g, r = cv2.split(frame)  # get b,g,r
                frame = cv2.merge([r, g, b])  # switch it to rgb
                ax.imshow(frame, aspect='equal')
                plt.axis('off')

                robot_name = ['red', 'blue', 'green', 'orange']

                # Draw trajectory to frame
                t_step = ExpSetup.LiDAR_RATE
                if current_step > 0:
                    current_datastep = min(int((current_step / frame_per_second) * data_freq), __end_idx)
                    min_data = max(current_datastep - (past_t[idx] * data_freq), 0)
                    # min_data = max(past_t[idx] * data_freq, 0)
                    for i in range(SceneSetup.robot_num):
                        px_key, py_key = 'pos_x_' + str(i), 'pos_y_' + str(i)
                        ax.scatter(pos_pxl[px_key][min_data:current_datastep:t_step],
                                   pos_pxl[py_key][min_data:current_datastep:t_step], s=1,
                                   color=SceneSetup.robot_color[i], label='Trajectory ' + robot_name[i] + ' robot' if idx == len(time_snap) - 1 else None)

                goal_name = ['Checkpoint red robot',
                             'Checkpoint blue robot',
                             'Checkpoint green robot',
                             'Checkpoint orange robot']
                for i in range(SceneSetup.robot_num):
                    px_key, py_key = 'pos_x_' + str(i), 'pos_y_' + str(i)
                    # pxl_goal_x, pxl_goal_y = pos_pxl[px_key][goal_snap[idx] * data_freq], \
                    #     pos_pxl[py_key][goal_snap[idx] * data_freq]
                    # print(*SceneSetup.goal_poses[i][goal_snap])
                    pxl_goal_x, pxl_goal_y = NebolabSetup.pos_m2pxl(*SceneSetup.goal_poses[i][goal_snap[idx]][:2])
                    ax.scatter(pxl_goal_x, pxl_goal_y, 50, marker='o', color=SceneSetup.robot_color[i],
                               edgecolors='black', label=goal_name[i])

                leg_size = 7
                if idx == len(time_snap) - 1: ax.legend(loc='lower right', prop={'size': leg_size})

                name = outname + str(snap_point) + '.pdf'
                pngname = outname + str(snap_point) + '.png'
                # plt.savefig(name, bbox_inches="tight", pad_inches=0, dpi=300)
                plt.savefig(pngname, bbox_inches="tight", pad_inches=0, dpi=300)
                print(pngname)


        if beautify:
            red = [(3, 30, 170), (15, 230, 255), (0, 0, 255)]
            blue = [(90, 90, 0), (160, 255, 255), (255, 0, 0)]
            green = [(75, 100, 50), (90, 200, 255), (0, 255, 0)]
            yellow = [(20, 100, 100), (50, 255, 255), (0, 255, 255)]
            colors = [red, blue, green, yellow]

            centers = [(0, 0) for _ in range(4)]
            drawn_comm_lines = {}

            out = cv2.VideoWriter(SimSetup.sim_defname + '_fixed.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                                  (int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            max_thickness = 2 * SceneSetup.max_form_epsilon.max()
            ratio = visData['last_idx'] / length
            while True:
                ret, frame = cam.read()
                if current_step < 0:
                    current_step += 1
                    continue
                if ret:
                    current_step += 1
                    print(f"Current step: {current_step:>5}", end="\r")
                    # cv2.imshow('frame', frame)
                    for i in range(SceneSetup.robot_num):
                        centers[i] = NebolabSetup.pos_m2pxl(__stored_data[f'pos_x_{i}'][int(current_step * ratio)],
                                                            __stored_data[f'pos_y_{i}'][int(current_step * ratio)])
                        # mask = prep(frame, colors[i][0], colors[i][1])
                        # center = contour(frame, mask, colors[i][2])
                        # if centers[i] == (0, 0):
                        #     if center is None: pass
                        #     else: centers[i] = center
                        # else:
                        #     if center is None: pass
                        #     else:
                        #         centers[i] = (int(centers[i][0] * 0.9 + center[0] * 0.1),
                        #                       int(centers[i][1] * 0.9 + center[1] * 0.1))
                        # cv2.circle(frame, centers[i], 10, colors[i][2], -1)
                        # out.write(frame)
                    for i in range(SceneSetup.robot_num):
                        for j in range(SceneSetup.robot_num):
                            if (i < j) and (SceneSetup.form_A[i, j] > 0):
                                cv2.line(frame, centers[i], centers[j], (0, 0, 0),
                                         int(max_thickness / (SceneSetup.max_form_epsilon[i][j] ** 0.8)))
                    if current_step % (length / cam.get(cv2.CAP_PROP_FRAME_COUNT)) == 0:
                        cv2.imshow('frame', frame)
                        out.write(frame)
                        key = cv2.waitKey(1) & 0xFF == ord('q')

                else:
                    break
            out.release()
        cam.release()
        cv2.destroyAllWindows()

def exp_pkl_plot():
    """
    Plot the logged data
    """
    # ---------------------------------------------------
    with open(SimSetup.sim_fdata_vis, 'rb') as f:
        visData = pickle.load(f)
    __stored_data = visData['stored_data']
    __end_idx = visData['last_idx']
    __end_idx = len([t for t in __stored_data["time"] if t is not None])

    # set __end_idx manually
    t_stop = SimSetup.tmax
    __end_idx = t_stop * ExpSetup.ROS_RATE - 1
    # __end_idx = t_stop * ExpSetup.LiDAR_RATE - 1
    # print(SceneSetup.robot_color)

    # print(__stored_data['time'])
    # Print all key datas
    print(len([t for t in __stored_data["time"] if t is not None]), __end_idx)
    print(
        f'The file {SimSetup.sim_fdata_vis} contains the following logs for {__stored_data["time"][__end_idx]:.2f} s:')
    print(__stored_data.keys())
    # ---------------------------------------------------
    figure_short = (6.4, 3.4)
    figure_size = (6.4, 4.8)
    FS = 16  # font size
    LW = 1.5  # line width
    leg_size = 8

    bias_robot_name = ['red robot', 'blue robot (biased)', 'green robot (biased)']
    robot_name = ['red robot', 'blue robot', 'green robot']

    # PLOT THE POSITION
    # ---------------------------------------------------¨
    num_goal = SceneSetup.goal_pos.shape[0]
    fig, ax = plt.subplots(2, figsize=figure_short)
    plt.rcParams.update({'font.size': FS})
    # plt.rcParams['text.usetex'] = True
    # plot
    plot_pickle_log_time_series_batch_robotid(ax[0], __stored_data, __end_idx, 'pos_x_')
    plot_pickle_log_time_series_batch_robotid(ax[1], __stored_data, __end_idx, 'pos_y_')
    # plot goal pos
    for goal_id in range(SceneSetup.goal_pos.shape[0]):
        ax[0].plot([0, t_stop], [SceneSetup.goal_pos[goal_id, 0], SceneSetup.goal_pos[goal_id, 0]], ':', color='k',
                   label='goal position')
        ax[1].plot([0, t_stop], [SceneSetup.goal_pos[goal_id, 1], SceneSetup.goal_pos[goal_id, 1]], ':', color='k',
                   label='goal position')
    # label
    ax[0].set(ylabel='X-position [m]')
    ax[1].set(xlabel="t [s]", ylabel='Y-position [m]')
    ax[0].legend(loc='best', prop={'size': leg_size})
    # plt.show()
    figname = SimSetup.sim_defname + '_pos.pdf'
    pngname = SimSetup.sim_defname + '_pos.png'
    # plt.savefig(figname, bbox_inches="tight", dpi=300)
    plt.savefig(pngname, bbox_inches="tight", dpi=300)
    print('export figure: ' + pngname, flush=True)

    # PLOT THE VELOCITY
    # ---------------------------------------------------
    fig, ax = plt.subplots(2, figsize=figure_short)
    plt.rcParams.update({'font.size': FS})
    # plt.rcParams['text.usetex'] = True
    # plot
    plot_pickle_log_time_series_batch_robotid(ax[0], __stored_data, __end_idx, 'u_nom_x_')
    plot_pickle_log_time_series_batch_robotid(ax[1], __stored_data, __end_idx, 'u_nom_y_')
    # label
    ax[0].set(ylabel='u_x [m/s]')
    ax[1].set(xlabel="t [s]", ylabel='u_y [m/s]')
    ax[1].legend(loc='best', prop={'size': leg_size})
    # plt.show()
    # figname = SimSetup.sim_defname + '_u_nom.pdf'
    pngname = SimSetup.sim_defname + '_u_nom.png'
    plt.savefig(pngname, bbox_inches="tight", dpi=300)
    print('export figure: ' + pngname, flush=True)

    # # PLOT THE DETECTION
    # # ---------------------------------------------------
    fig = plt.figure(figsize=figure_short)
    plt.rcParams.update({'font.size': FS})
    # plt.rcParams['text.usetex'] = True
    ax = plt.gca()
    # plot
    plot_pickle_log_time_series_batch_keys(ax, __stored_data, __end_idx, 'h_staticobs_')
    ax.set(xlabel="t [s]", ylabel='h LiDAR')
    ax.legend(loc='best', prop={'size': leg_size})
    # plt.show()
    pngname = SimSetup.sim_defname + '_staticobs.png'
    plt.savefig(pngname, bbox_inches="tight", dpi=300)
    print('export figure: ' + pngname, flush=True)

    # # PLOT THE DETECTION
    # # ---------------------------------------------------
    fig = plt.figure(figsize=(6.4, 2.0))
    plt.rcParams.update({'font.size': FS})
    # plt.rcParams['text.usetex'] = True
    ax = plt.gca()
    # plot
    ## This is for experiment
    # __stored_data['h_lidar_1'] = [(v + SceneSetup.d_obs ** 2) ** 0.5 if v is not None else v for v in __stored_data['h_staticobs_0']][25:]
    # __stored_data['h_lidar_2'] = [(v + SceneSetup.d_obs ** 2) ** 0.5 if v is not None else v for v in __stored_data['h_staticobs_1']][25:]
    # __stored_data['h_lidar_3'] = [(v + SceneSetup.d_obs ** 2) ** 0.5 if v is not None else v for v in __stored_data['h_staticobs_2']][25:]
    # __stored_data['h_lidar_4'] = [(v + SceneSetup.d_obs ** 2) ** 0.5 if v is not None else v for v in __stored_data['h_staticobs_3']][25:]
    # This is for simulation
    __stored_data['lidar_0'][:5] = [v for v in __stored_data['lidar_0']][5:10]
    __stored_data['lidar_1'][:5] = [v for v in __stored_data['lidar_1']][5:10]
    __stored_data['lidar_2'][:5] = [v for v in __stored_data['lidar_2']][5:10]
    __stored_data['lidar_3'][:5] = [v for v in __stored_data['lidar_3']][5:10]
    # Swap variables to get start index at 1
    __stored_data['lidar_4'] = __stored_data['lidar_3']
    __stored_data['lidar_3'] = __stored_data['lidar_2']
    __stored_data['lidar_2'] = __stored_data['lidar_1']
    __stored_data['lidar_1'] = __stored_data['lidar_0']
    del __stored_data['lidar_0']
    plot_pickle_log_time_series_batch_keys(ax, __stored_data, __end_idx, 'lidar_')
    # plot_pickle_log_time_series_batch_keys(ax, __stored_data, __end_idx, 'h_lidar_')
    ax.plot(__stored_data['time'][:__end_idx], [SceneSetup.d_obs for _ in range(__end_idx)], linestyle='dashed', color='k', label='$R_s$')
    ax.set(xlabel="t [s]", ylabel='LiDAR')
    ax.legend(loc='best', prop={'size': leg_size})
    # plt.show()
    pngname = SimSetup.sim_defname + '_lidar.png'
    plt.savefig(pngname, bbox_inches="tight", dpi=300)
    print('export figure: ' + pngname, flush=True)

    # # PLOT THE EPS
    # # ---------------------------------------------------
    if SceneSetup.USECBF_FORMATION:
        fig = plt.figure(figsize=figure_short)
        plt.rcParams.update({'font.size': FS})
        # plt.rcParams['text.usetex'] = True
        ax = plt.gca()
        # plot
        plot_pickle_log_time_series_batch_keys(ax, __stored_data, __end_idx, 'eps_')
        ax.set(xlabel="t [s]", ylabel='distance [m]')
        ax.legend(loc='best', prop={'size': leg_size})
        # plt.show()
        pngname = SimSetup.sim_defname + '_eps.png'
        plt.savefig(pngname, bbox_inches="tight", dpi=300)
        print('export figure: ' + pngname, flush=True)

        # # PLOT THE H EPS CEIL
        # # ---------------------------------------------------
        fig = plt.figure(figsize=figure_short)
        plt.rcParams.update({'font.size': FS})
        # plt.rcParams['text.usetex'] = True
        ax = plt.gca()
        # plot
        plot_pickle_log_time_series_batch_keys(ax, __stored_data, __end_idx, 'h_eps_ceil_')
        ax.set(xlabel="t [s]", ylabel="h_eps_ceil")
        ax.legend(loc='best', prop={'size': leg_size})
        # plt.show()
        pngname = SimSetup.sim_defname + '_h_eps_ceil.png'
        plt.savefig(pngname, bbox_inches="tight", dpi=300)
        print('export figure: ' + pngname, flush=True)

        # # PLOT THE H EPS FLOOR
        # # ---------------------------------------------------
        fig = plt.figure(figsize=figure_short)
        plt.rcParams.update({'font.size': FS})
        # plt.rcParams['text.usetex'] = True
        ax = plt.gca()
        # plot
        plot_pickle_log_time_series_batch_keys(ax, __stored_data, __end_idx, 'h_eps_floor_')
        ax.set(xlabel="t [s]", ylabel="h_eps_floor")
        ax.legend(loc='best', prop={'size': leg_size})
        # plt.show()
        pngname = SimSetup.sim_defname + '_h_eps_floor.png'
        plt.savefig(pngname, bbox_inches="tight", dpi=300)
        print('export figure: ' + pngname, flush=True)

        # # PLOT THE H FMU
        # # ---------------------------------------------------
        fig = plt.figure(figsize=figure_short)
        plt.rcParams.update({'font.size': FS})
        # plt.rcParams['text.usetex'] = True
        ax = plt.gca()
        # plot
        plot_pickle_log_time_series_batch_keys(ax, __stored_data, __end_idx, 'h_fmu_')
        ax.set(xlabel="t [s]", ylabel="h_fmu")
        ax.legend(loc='best', prop={'size': leg_size})
        # plt.show()
        pngname = SimSetup.sim_defname + '_h_fmu.png'
        plt.savefig(pngname, bbox_inches="tight", dpi=300)
        print('export figure: ' + pngname, flush=True)

        # # # PLOT THE FORMATION DISTANCE
        # # # This segment is customized
        # # # TODO: Generic approach
        # # # ---------------------------------------------------
        ml = min([len(__stored_data['pos_x_0']), len(__stored_data['pos_x_2']),
                  len(__stored_data['pos_x_1']), len(__stored_data['pos_x_0']), len(__stored_data['pos_x_0'])])
        # Timeseries Desired Distance
        __stored_data['Rs'] = [SceneSetup.d_obs for idx in range(len(__stored_data['pos_x_0'])) if __stored_data['pos_x_0'][idx] is not None]
        __stored_data['disD_0_1'] = [SceneSetup.form_A[0, 1] for idx in range(ml) if __stored_data['pos_x_0'][idx] is not None]
        __stored_data['disD_2_3'] = [SceneSetup.form_A[2, 3] for idx in range(ml) if __stored_data['pos_x_2'][idx] is not None]
        __stored_data['disD_1_2'] = [SceneSetup.form_A[1, 2] for idx in range(ml) if __stored_data['pos_x_1'][idx] is not None]
        __stored_data['disD_0_3'] = [SceneSetup.form_A[0, 3] for idx in range(ml) if __stored_data['pos_x_0'][idx] is not None]
        __stored_data['disD_0_2'] = [SceneSetup.form_A[0, 2] for idx in range(ml) if __stored_data['pos_x_0'][idx] is not None]

        # Data Length
        length = min([len(__stored_data['disD_0_1']), len(__stored_data['disD_2_3']), len(__stored_data['disD_1_2']), len(__stored_data['disD_0_3']), len(__stored_data['disD_0_2'])])

        dist = lambda  data, i, j: [((data[f'pos_x_{i}'][idx] - data[f'pos_x_{j}'][idx]) ** 2 +
                                      (data[f'pos_y_{j}'][idx] - data[f'pos_y_{i}'][idx]) ** 2) ** 0.5
                                      for idx in range(len(data[f'pos_x_{i}'])) if data[f'pos_x_{i}'][idx] is not None]
        disu = lambda data, i, j: [data[f'disD_{i}_{j}'][idx] + data[f'eps_{i}_{j}'][idx] + data[f'eps_{j}_{i}'][idx]
                                   for idx in range(length)]
        disl = lambda data, i, j: [max(data[f'disD_{i}_{j}'][idx] - data[f'eps_{i}_{j}'][idx] - data[f'eps_{j}_{i}'][idx], 0)
                                   for idx in range(length)]
        disU = lambda data, i, j: [data[f'disD_{i}_{j}'][idx] + SceneSetup.max_form_epsilon[i, j]
                                   for idx in range(length)]
        disL = lambda data, i, j: [max(data[f'disD_{i}_{j}'][idx] - SceneSetup.max_form_epsilon[i, j], 0)
                                   for idx in range(length)]

        __stored_data['dist_0_1'] = dist(__stored_data, 0, 1)
        __stored_data['dist_2_3'] = dist(__stored_data, 2, 3)
        __stored_data['dist_1_2'] = dist(__stored_data, 1, 2)
        __stored_data['dist_0_3'] = dist(__stored_data, 0, 3)
        __stored_data['dist_0_2'] = dist(__stored_data, 0, 2)

        __stored_data['disu_0_1'] = disu(__stored_data, 0, 1)
        __stored_data['disl_0_1'] = disl(__stored_data, 0, 1)
        __stored_data['disU_0_1'] = disU(__stored_data, 0, 1)
        __stored_data['disL_0_1'] = disL(__stored_data, 0, 1)

        __stored_data['disu_2_3'] = disu(__stored_data, 2, 3)
        __stored_data['disl_2_3'] = disl(__stored_data, 2, 3)
        __stored_data['disU_2_3'] = disU(__stored_data, 2, 3)
        __stored_data['disL_2_3'] = disL(__stored_data, 2, 3)

        __stored_data['disu_1_2'] = disu(__stored_data, 1, 2)
        __stored_data['disl_1_2'] = disl(__stored_data, 1, 2)
        __stored_data['disU_1_2'] = disU(__stored_data, 1, 2)
        __stored_data['disL_1_2'] = disL(__stored_data, 1, 2)

        __stored_data['disu_0_3'] = disu(__stored_data, 0, 3)
        __stored_data['disl_0_3'] = disl(__stored_data, 0, 3)
        __stored_data['disU_0_3'] = disU(__stored_data, 0, 3)
        __stored_data['disL_0_3'] = disL(__stored_data, 0, 3)

        __stored_data['disu_0_2'] = disu(__stored_data, 0, 2)
        __stored_data['disl_0_2'] = disl(__stored_data, 0, 2)
        __stored_data['disU_0_2'] = disU(__stored_data, 0, 2)
        __stored_data['disL_0_2'] = disL(__stored_data, 0, 2)

        __end_idx = len(__stored_data['disD_0_1'])

        # PLOTTING
        # Customized: 05 data to plot
        # TODO: generic approach
        fig, ax = plt.subplots(nrows=5, ncols=1, sharex=True, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 1, 2, 2, 4]})
        plt.rcParams.update({'font.size': FS})

        # Subplot 1
        plot_pickle_log_time_series_batch_keys(ax[0], __stored_data, __end_idx, 'dist_0_1')
        ax[0].fill_between(__stored_data['time'][:__end_idx], __stored_data['disl_0_1'], __stored_data['disu_0_1'],
                           alpha=0.2, color='k', linewidth=0)
        ax[0].plot(__stored_data['time'][:__end_idx], __stored_data['disD_0_1'], color='k', linestyle='dashed', linewidth=1)
        ax[0].plot(__stored_data['time'][:__end_idx], __stored_data['disU_0_1'], color='r', linestyle='dashed', linewidth=1)
        ax[0].plot(__stored_data['time'][:__end_idx], __stored_data['disL_0_1'], color='r', linestyle='dashed', linewidth=1)
        ax[0].set(ylim=(__stored_data['disL_0_1'][0] - 0.1, __stored_data['disU_0_1'][0] + 0.1))

        # Subplot 2
        plot_pickle_log_time_series_batch_keys(ax[1], __stored_data, __end_idx, 'dist_2_3')
        ax[1].fill_between(__stored_data['time'][:__end_idx], __stored_data['disl_2_3'], __stored_data['disu_2_3'],
                           alpha=0.2, color='k', linewidth=0)
        ax[1].plot(__stored_data['time'][:__end_idx], __stored_data['disD_2_3'], color='k', linestyle='dashed',
                   linewidth=1)
        ax[1].plot(__stored_data['time'][:__end_idx], __stored_data['disU_2_3'], color='r', linestyle='dashed',
                   linewidth=1)
        ax[1].plot(__stored_data['time'][:__end_idx], __stored_data['disL_2_3'], color='r', linestyle='dashed',
                   linewidth=1)
        ax[1].set(ylim=(__stored_data['disL_2_3'][0] - 0.1, __stored_data['disU_2_3'][0] + 0.1))

        # Subplot 3
        plot_pickle_log_time_series_batch_keys(ax[2], __stored_data, __end_idx, 'dist_1_2')
        ax[2].fill_between(__stored_data['time'][:__end_idx], __stored_data['disl_1_2'], __stored_data['disu_1_2'],
                           alpha=0.2, color='k', linewidth=0)
        ax[2].plot(__stored_data['time'][:__end_idx], __stored_data['disD_1_2'], color='k', linestyle='dashed',
                   linewidth=1)
        ax[2].plot(__stored_data['time'][:__end_idx], __stored_data['disU_1_2'], color='r', linestyle='dashed',
                   linewidth=1)
        ax[2].plot(__stored_data['time'][:__end_idx], __stored_data['disL_1_2'], color='r', linestyle='dashed',
                   linewidth=1)
        ax[2].set(ylim=(__stored_data['disL_1_2'][0] - 0.1, __stored_data['disU_1_2'][0] + 0.1))

        # Subplot 4
        plot_pickle_log_time_series_batch_keys(ax[3], __stored_data, __end_idx, 'dist_0_3')
        ax[3].fill_between(__stored_data['time'][:__end_idx], __stored_data['disl_0_3'], __stored_data['disu_0_3'],
                           alpha=0.2, color='k', linewidth=0)
        ax[3].plot(__stored_data['time'][:__end_idx], __stored_data['disD_0_3'], color='k', linestyle='dashed',
                   linewidth=1)
        ax[3].plot(__stored_data['time'][:__end_idx], __stored_data['disU_0_3'], color='r', linestyle='dashed',
                   linewidth=1)
        ax[3].plot(__stored_data['time'][:__end_idx], __stored_data['disL_0_3'], color='r', linestyle='dashed',
                   linewidth=1)
        ax[3].set(ylim=(__stored_data['disL_0_3'][0] - 0.1, __stored_data['disU_0_3'][0] + 0.1))

        # Subplot 5
        plot_pickle_log_time_series_batch_keys(ax[4], __stored_data, __end_idx, 'dist_0_2')
        ax[4].fill_between(__stored_data['time'][:__end_idx], __stored_data['disl_0_2'], __stored_data['disu_0_2'],
                           alpha=0.2, color='k', linewidth=0)
        ax[4].plot(__stored_data['time'][:__end_idx], __stored_data['disD_0_2'], color='k', linestyle='dashed',
                   linewidth=1)
        ax[4].plot(__stored_data['time'][:__end_idx], __stored_data['disU_0_2'], color='r', linestyle='dashed',
                   linewidth=1)
        ax[4].plot(__stored_data['time'][:__end_idx], __stored_data['disL_0_2'], color='r', linestyle='dashed',
                   linewidth=1)
        ax[4].set(ylim=(__stored_data['disL_0_2'][0] - 0.1, __stored_data['disU_0_2'][0] + 0.1))

        # Axes title
        ax[0].set_ylabel("$\Vert p_1-p_2 \Vert$", fontsize=10)
        ax[1].set_ylabel("$\Vert p_3-p_4 \Vert$", fontsize=10)
        ax[2].set_ylabel("$\Vert p_2-p_3 \Vert$", fontsize=10)
        ax[3].set_ylabel("$\Vert p_1-p_4 \Vert$", fontsize=10)
        ax[4].set_ylabel("$\Vert p_1-p_3 \Vert$", fontsize=10)
        ax[4].set_xlabel("t [s]")

        # Save
        pngname = SimSetup.sim_defname + '_form_dist.png'
        plt.savefig(pngname, bbox_inches='tight',dpi=300)
        print('export figure: ' + pngname, flush=True)

    # # PLOT THE H STATICOBS
    # # ---------------------------------------------------
    if SceneSetup.USECBF_STATICOBS:
        fig = plt.figure(figsize=figure_short)
        plt.rcParams.update({'font.size': FS})
        # plt.rcParams['text.usetex'] = True
        ax = plt.gca()
        # plot
        plot_pickle_log_time_series_batch_keys(ax, __stored_data, __end_idx, 'h_staticobs_')
        ax.set(xlabel="t [s]", ylabel="h_staticobs")
        ax.legend(loc='best', prop={'size': leg_size})
        # plt.show()
        pngname = SimSetup.sim_defname + '_h_staticobs.png'
        plt.savefig(pngname, bbox_inches="tight", dpi=300)
        print('export figure: ' + pngname, flush=True)

    plt.close('all')

if __name__ == '__main__':
    import sys

    preamble_setting(sys.argv[1])
    # scenario_pkl_plot()
    # exp_video_pkl_plot(snap=True)
    # exp_video_pkl_plot(beautify=True)
    exp_pkl_plot()