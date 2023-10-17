import pickle
import matplotlib.pyplot as plt

from scenarios_unicycle.CCTA2024_Controller import SceneSetup, np
from scenarios_unicycle.CCTA2024_FormationObstacleLidar_scenario import SimSetup, ExpSetup


# def preamble_setting():  # when manual plotting is needed
#     """
#
#     :return:
#     """
#     # List of scenario mode
#     SceneSetup.SCENARIO_MODE = 0  # basic leader-following no attack
#     # SceneSetup.SCENARIO_MODE = 1 # with attack but no defense
#     # SceneSetup.SCENARIO_MODE = 2 # with attack and with defense
#     # SceneSetup.SCENARIO_MODE = 3 # with attack and another version of defense
#
#     # SimSetup.sim_defname = 'animation_result/Resilient_scenario/sim_' + str(SceneSetup.SCENARIO_MODE)
#     # SimSetup.sim_fdata_log = SimSetup.sim_defname + '_vis.pkl'
#
#     # Temporary fix for experiment data TODO: fix this later
#     SimSetup.sim_defname = 'experiment_result/Resilient_scenario/exp_' + str(SceneSetup.SCENARIO_MODE)  # + '_beta2'
#     SimSetup.sim_fdata_log = SimSetup.sim_defname + '_data.pkl'


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
        ax.plot(time_data, key_data, label=key.strip(pre_string))
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
    __end_idx = visData['last_idx']

    # set __end_idx manually
    t_stop = SimSetup.tmax
    __end_idx = t_stop * ExpSetup.ROS_RATE - 1

    # print(__stored_data['time'])
    # Print all key datas
    print(f'The file {SimSetup.sim_fdata_vis} contains the following logs for {__stored_data["time"][__end_idx]:.2f} s:')
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

    # # PLOT THE H STATIC_FF_CIRCLE
    # # ---------------------------------------------------
    if SceneSetup.USECBF_STATIC_FF_CIRCLE:
        fig = plt.figure(figsize=figure_short)
        plt.rcParams.update({'font.size': FS})
        # plt.rcParams['text.usetex'] = True
        ax = plt.gca()
        # plot
        plot_pickle_log_time_series_batch_keys(ax, __stored_data, __end_idx, 'h_static_ffc_')
        ax.set(xlabel="t [s]", ylabel="h_static_ffc")
        ax.legend(loc='best', prop={'size': leg_size})
        # plt.show()
        pngname = SimSetup.sim_defname + '_h_static_ffc.png'
        plt.savefig(pngname, bbox_inches="tight", dpi=300)
        print('export figure: ' + pngname, flush=True)

    if SceneSetup.USECBF_ELLIPSEAV:
        fig = plt.figure(figsize=figure_short)
        plt.rcParams.update({'font.size': FS})
        # plt.rcParams['text.usetex'] = True
        ax = plt.gca()
        # plot
        plot_pickle_log_time_series_batch_keys(ax, __stored_data, __end_idx, 'h_ellipseav_')
        ax.set(xlabel="t [s]", ylabel="h_ellipseav")
        ax.legend(loc='best', prop={'size': leg_size})
        # plt.show()
        pngname = SimSetup.sim_defname + '_h_ellipseav.png'
        plt.savefig(pngname, bbox_inches="tight", dpi=300)
        print('export figure: ' + pngname, flush=True)

    # PLOT THE DISTANCE
    # ---------------------------------------------------
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


def exp_video_pkl_plot():
    """

    :return:
    """
    # videoloc = None

    videoloc = SimSetup.sim_defname + '.avi'  # TODO: fix the variable loc
    outname = SimSetup.sim_defname + 'snap_'
    if SceneSetup.SCENARIO_MODE < 2:
        time_snap = [20]  # in seconds
    else:
        time_snap = [62]  # in seconds
    past_t = time_snap[0]  # in seconds
    frame_shift = 0  # accomodate on-sync video and data, video ALWAYS earlier
    data_freq = ExpSetup.ROS_RATE

    if videoloc is not None:
        import cv2
        from nebolab_experiment_setup import NebolabSetup

        # Initialize VIDEO
        cam = cv2.VideoCapture(videoloc)
        frame_per_second = cam.get(cv2.CAP_PROP_FPS)

        current_step = -frame_shift

        # Initialize Pickle
        with open(SimSetup.sim_fdata_log, 'rb') as f:
            visData = pickle.load(f)
        __stored_data = visData['stored_data']
        __end_idx = visData['last_idx']
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

        # Proceed with video reading
        for snap_point in time_snap:
            # roll the video towards snap_point
            while current_step <= snap_point * frame_per_second:
                ret, frame = cam.read()
                if ret:
                    current_step += 1
                else:
                    break

            # plot on snap_point
            fig = plt.figure
            ax = plt.gca()

            b, g, r = cv2.split(frame)  # get b,g,r
            frame = cv2.merge([r, g, b])  # switch it to rgb
            ax.imshow(frame, aspect='equal')
            plt.axis('off')

            robot_name = ['red', 'blue', 'green']

            # Draw trajectory to frame
            t_step = int(ExpSetup.ROS_RATE / 2)
            if current_step > 0:
                current_datastep = min(int((current_step / frame_per_second) * data_freq), __end_idx)
                min_data = max(current_datastep - (past_t * data_freq), 0)
                for i in range(SceneSetup.robot_num):
                    px_key, py_key = 'pos_x_' + str(i), 'pos_y_' + str(i)
                    ax.scatter(pos_pxl[px_key][min_data:current_datastep:t_step],
                               pos_pxl[py_key][min_data:current_datastep:t_step],
                               3, color=SceneSetup.robot_color[i], label='Trajectory ' + robot_name[i] + ' robot')

            goal_name = ['Goal position', 'Biased goal (blue robot)', 'Biased goal (green robot)']
            for i in range(SceneSetup.robot_num):
                goal_i = SceneSetup.goal_pos[0] - SceneSetup.pos_shift[i]
                pxl_goal_x, pxl_goal_y = NebolabSetup.pos_m2pxl(goal_i[0], goal_i[1])
                ax.scatter(pxl_goal_x, pxl_goal_y, 50, marker='*', color=SceneSetup.robot_color[i],
                           edgecolors='black', label=goal_name[i])

            leg_size = 7
            ax.legend(loc='center left', prop={'size': leg_size})

            name = outname + str(snap_point) + '.pdf'
            pngname = outname + str(snap_point) + '.png'
            plt.savefig(name, bbox_inches="tight", pad_inches=0, dpi=300)
            plt.savefig(pngname, bbox_inches="tight", pad_inches=0, dpi=300)
            print(name)

        cam.release()
        cv2.destroyAllWindows()
