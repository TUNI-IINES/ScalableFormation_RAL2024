import pickle
import numpy as np
import matplotlib.pyplot as plt

from scenarios_unicycle.CCTA2024_Controller import FeedbackInformation, SceneSetup, np
from scenarios_unicycle.CCTA2024_FormationObstacleLidar_scenario import SimSetup, ExpSetup, import_scenario

import cv2
from nebolab_experiment_setup import NebolabSetup


# FILE SETTINGS
DEF_FOLDER = 'experiment_result/20240418/'
FNAME = 'formation4_exp'

# FIGURE SETTINGS
figure_short = (6.4, 3.4)
figure_size = (6.4, 4.8)
FS, LW = 16, 1.5 # font size and line width
leg_size = 8


def preamble_setting():  # when manual plotting is needed
    # Temporary fix for animation or experiment data TODO: fix this later
    import_scenario(directory=DEF_FOLDER, filename=FNAME)
    SimSetup.sim_defname = DEF_FOLDER + FNAME
    SimSetup.sim_fdata_vis = SimSetup.sim_defname + '.pkl'


def scenario_pkl_plot():
    """
    Plot the logged experiment data
    """
    # ---------------------------------------------------
    # READING THE PICKLED DATA
    # ---------------------------------------------------
    with open(SimSetup.sim_fdata_vis, 'rb') as f: visData = pickle.load(f)
    __stored_data = visData['stored_data']
    __end_idx = visData['last_idx']

    # set __end_idx manually
    # t_stop = 62
    # __end_idx = t_stop * ExpSetup.ROS_RATE

    # print(__stored_data['time'])
    # Print all key datas
    print('The file ' + SimSetup.sim_fdata_vis + ' contains the following logs for ' + '{:.2f}'.format(__stored_data['time'][__end_idx]) + ' s:') 
    print(__stored_data.keys())

    # PLOT ALL DATA FOR DEBUG/CHECKING PURPOSES
    plot_XY_position(__stored_data, __end_idx)
    plot_nominal_velocity(__stored_data, __end_idx)
    plot_rectified_velocity(__stored_data, __end_idx)

    plot_cbf_avoidance(__stored_data, __end_idx)
    plot_cbf_formation(__stored_data, __end_idx)

    # FOR JOURNAL
    plot_distance_to_obs(__stored_data, __end_idx, start_idx=90)
    plot_distance_formation(__stored_data, __end_idx)


def exp_video_pkl_plot():
    videoloc = DEF_FOLDER + FNAME + '.avi'
    outname = DEF_FOLDER + 'snap_'

    # NOTE: the time is based on the video time. Which at the moment is a bit unsync. 
    # time_snap = [10, 60, 140, 240, 330, 413]  # in seconds
    time_snap = [0, 16, 32, 48, 64, 80]  # in seconds

    frame_shift = 0  # accomodate on-sync video and data, video ALWAYS earlier
    data_shift = int(5.8*ExpSetup.ROS_RATE) # accomodate delayed data (slow computation)

    # Generate BGR color
    bgr_color = {}
    for i in range(SceneSetup.robot_num):
        h = SceneSetup.robot_color[i].lstrip('#')
        r, g, b = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        bgr_color[i] = (b, g, r)


    if videoloc is not None:
        # -------------------------------
        # VIDEO DATA
        # -------------------------------
        # Initialize VIDEO
        cam = cv2.VideoCapture(videoloc)
        frame_per_second = cam.get(cv2.CAP_PROP_FPS)
        frame_count = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
        width  = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
        height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`


        # -------------------------------
        # RECORDED PICKLE DATA - for plotting waypoints only
        # -------------------------------
        # Initialize Pickle
        with open(SimSetup.sim_fdata_vis, 'rb') as f:
            visData = pickle.load(f)
            print(visData['stored_data'].keys())
        __stored_data = visData['stored_data']
        __end_idx = visData['last_idx']

        goal_pxl = {i:np.zeros(2) for i in range(SceneSetup.robot_num) }

        print('Frames:', frame_count, ", Time:", visData['last_idx'])
        # print('Keys', __stored_data.keys())
        # SceneSetup.robot_color = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e']

        # -------------------------------
        # IMAGE-BASED LOCALIZATION - for plotting trajectory
        # -------------------------------
        # Template for position 
        pos_pxl = {i:np.zeros((frame_count, 2)) for i in range(SceneSetup.robot_num) }
        # Initialized localization
        from experiment_result.camerabased_localization import localize_from_ceiling
        localizer = localize_from_ceiling()

        # -------------------------------
        # VIDEO OUTPUT
        # -------------------------------
        out = cv2.VideoWriter(SimSetup.sim_defname + '_fixed.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), frame_per_second, (width, height))
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions

        # -------------------------------
        # MAIN-LOOP OVER THE VIDEO FRAME
        # -------------------------------
        current_step = -frame_shift # syncing with the start of pickle data

        snap_iter = 0
        snap_timing_step = time_snap[snap_iter]*frame_per_second
        past_snap_step = 0 

        while True:
            # READ each frame
            ret, frame = cam.read()

            # ROLL frame until current_step = 0
            if current_step < 0:
                current_step += 1
                continue

            if ret:
                # Save position data from frame
                poses_center, poses_ahead = localizer.localize_all_robots(frame)
                frame = localizer.draw_pose(frame)

                for i in range(SceneSetup.robot_num):
                    is_valid_data = False
                    if poses_ahead[i] is not None:
                        pos_pxl[i][current_step,0], pos_pxl[i][current_step,1] = \
                            NebolabSetup.pos_m2pxl( poses_ahead[i].x, poses_ahead[i].y)
                        
                        # Error checking
                        if pos_pxl[i][current_step,0].is_integer(): 
                            # valid integer value, check distance from past data
                            dx = pos_pxl[i][current_step,0] - pos_pxl[i][current_step-1,0]
                            dy = pos_pxl[i][current_step,1] - pos_pxl[i][current_step-1,1]
                            dist = np.sqrt(dx**2 + dy**2)

                            # 0.01 m is the assumed max distance for 1 iteration in 30fps
                            is_valid_data = dist < 0.02 * NebolabSetup.SCALE_M2PXL
                        
                    # Invalid data, alternatively use last data
                    if current_step > 0 and not is_valid_data:
                        pos_pxl[i][current_step,0] = pos_pxl[i][current_step-1,0]
                        pos_pxl[i][current_step,1] = pos_pxl[i][current_step-1,1]

                # Get goal data from pickle
                time = current_step / frame_per_second
                idx = int(time * ExpSetup.ROS_RATE) + data_shift
                
                for i in range(SceneSetup.robot_num):
                    gx = __stored_data[f"goal_x_{i}"][idx]
                    gy = __stored_data[f"goal_y_{i}"][idx]
                    goal_pxl[i][0], goal_pxl[i][1] = NebolabSetup.skewed_pos_m2pxl(gx, gy)

                # Plot image to frame
                augment_video_frame(frame, current_step, pos_pxl, goal_pxl, bgr_color)
                
                # save if snap
                if current_step > snap_timing_step:
                    
                    frame_snap = frame.copy()
                    snap_name = outname + str(time_snap[snap_iter]) + '.jpg'
                    generate_snap(frame_snap, current_step, past_snap_step, pos_pxl, bgr_color, snap_name)

                    # advance snap timing
                    past_snap_step = current_step
                    snap_iter += 1 
                    if snap_iter < len(time_snap):
                        snap_timing_step = time_snap[snap_iter]*frame_per_second
                    else:
                        snap_timing_step = frame_count + 1

                # save video
                cv2.imshow('frame', frame)
                key = cv2.waitKey(1) & 0xFF == ord('q')
                out.write(frame)

                # Advance step
                current_step += 1

            else:
                break
        
        out.release()
        cam.release()
        cv2.destroyAllWindows()

            

def augment_video_frame(frame, current_step, pos_pxl, goal_pxl, bgr_color):
    
    max_thickness = 8 * SceneSetup.max_form_epsilon.max()

    for i in range(SceneSetup.robot_num):
        # Draw formation line
        for j in range(SceneSetup.robot_num):
            if (i < j) and (SceneSetup.form_A[i, j] > 0):
                if (pos_pxl[i][current_step,0] is not None) and (pos_pxl[j][current_step,0] is not None):
                    pxl_i = ( int(pos_pxl[i][current_step,0]), int(pos_pxl[i][current_step,1]) )
                    pxl_j = ( int(pos_pxl[j][current_step,0]), int(pos_pxl[j][current_step,1]) )
                    cv2.line(frame, pxl_i, pxl_j, (0, 0, 0),
                                int(max_thickness / (SceneSetup.max_form_epsilon[i][j] ** 0.8)))
                
        # Draw waypoint        
        pxl_goal_i = ( int(goal_pxl[i][0]), int(goal_pxl[i][1]) )
        cv2.circle(frame, pxl_goal_i, 10, bgr_color[i], -1)
        cv2.circle(frame, pxl_goal_i, 12, (0,0,0), 4)


def generate_snap(frame, current_step, past_snap_step, pos_pxl, bgr_color, fname):

    line_width = 8

    for i in range(SceneSetup.robot_num):
        for step in range(past_snap_step+1, current_step):
            pxl_from = ( int(pos_pxl[i][step-1,0]), int(pos_pxl[i][step-1,1]) )
            pxl_to = ( int(pos_pxl[i][step,0]), int(pos_pxl[i][step,1]) )
            # draw line segment
            cv2.line(frame, pxl_from, pxl_to, bgr_color[i], line_width)

    cv2.imwrite(fname, frame)
    print('exporting snap: ' + fname, flush=True)


# ---------------------------------------------------
# SPECIFIC PLOTTING FUNCTIONS
# ---------------------------------------------------
def plot_XY_position(__stored_data, __end_idx):
    # PLOT THE POSITION
    # ---------------------------------------------------¨
    fig, ax = plt.subplots(2, figsize=figure_short)
    plt.rcParams.update({'font.size': FS})
    # plt.rcParams['text.usetex'] = True
    # plot
    plot_pickle_log_time_series_batch_robotid(ax[0], __stored_data, __end_idx, 'pos_x_')
    plot_pickle_log_time_series_batch_robotid(ax[1], __stored_data, __end_idx, 'pos_y_')
    # plot goal pos
    time_data = __stored_data['time'][:__end_idx]
    for i in range(SceneSetup.robot_num):
        ax[0].plot(time_data, __stored_data[f"goal_x_{i}"][:__end_idx], ':', color=SceneSetup.robot_color[i])
        ax[1].plot(time_data, __stored_data[f"goal_y_{i}"][:__end_idx], ':', color=SceneSetup.robot_color[i])
    # label
    ax[0].set(ylabel='X-position [m]')
    ax[1].set(xlabel="t [s]", ylabel='Y-position [m]')
    ax[0].legend(loc='best', prop={'size': leg_size})
    # plt.show()
    # figname = SimSetup.sim_defname + '_pos.pdf'
    # plt.savefig(figname, bbox_inches="tight", dpi=300)
    pngname = SimSetup.sim_defname + '_pos.png'
    plt.savefig(pngname, bbox_inches="tight", dpi=300)
    print('export figure: ' + pngname, flush=True)


def plot_nominal_velocity(__stored_data, __end_idx):
    # PLOT THE VELOCITY
    # ---------------------------------------------------
    fig, ax = plt.subplots(2, figsize=figure_short)
    plt.rcParams.update({'font.size': FS})
    # plt.rcParams['text.usetex'] = True
    # plot
    plot_pickle_log_time_series_batch_robotid(ax[0], __stored_data, __end_idx, 'u_nom_x_')
    plot_pickle_log_time_series_batch_robotid(ax[1], __stored_data, __end_idx, 'u_nom_y_')
    # label
    ax[0].set(ylabel='u_x_nom [m/s]')
    ax[1].set(xlabel="t [s]", ylabel='u_y_nom [m/s]')
    ax[1].legend(loc='best', prop={'size': leg_size})
    # plt.show()
    # figname = SimSetup.sim_defname + '_u_nom.pdf'
    pngname = SimSetup.sim_defname + '_u_nom.png'
    plt.savefig(pngname, bbox_inches="tight", dpi=300)
    print('export figure: ' + pngname, flush=True)


def plot_rectified_velocity(__stored_data, __end_idx):
    # PLOT THE VELOCITY
    # ---------------------------------------------------
    fig, ax = plt.subplots(2, figsize=figure_short)
    plt.rcParams.update({'font.size': FS})
    # plt.rcParams['text.usetex'] = True
    # plot
    plot_pickle_log_time_series_batch_robotid(ax[0], __stored_data, __end_idx, 'u_x_')
    plot_pickle_log_time_series_batch_robotid(ax[1], __stored_data, __end_idx, 'u_y_')
    # label
    ax[0].set(ylabel='u_x [m/s]')
    ax[1].set(xlabel="t [s]", ylabel='u_y [m/s]')
    ax[1].legend(loc='best', prop={'size': leg_size})
    # plt.show()
    # figname = SimSetup.sim_defname + '_u_nom.pdf'
    pngname = SimSetup.sim_defname + '_u_rectified.png'
    plt.savefig(pngname, bbox_inches="tight", dpi=300)
    print('export figure: ' + pngname, flush=True)


def plot_cbf_avoidance(__stored_data, __end_idx):
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
    pngname = SimSetup.sim_defname + '_cbfavo_LiDAR.png'
    plt.savefig(pngname, bbox_inches="tight", dpi=300)
    print('export figure: ' + pngname, flush=True)


def plot_cbf_formation(__stored_data, __end_idx):
    # # PLOT THE EPSILON VALUE
    # # ---------------------------------------------------
    fig = plt.figure(figsize=figure_short)
    plt.rcParams.update({'font.size': FS})
    # plt.rcParams['text.usetex'] = True
    ax = plt.gca()
    # plot
    plot_pickle_log_time_series_batch_keys(ax, __stored_data, __end_idx, 'eps_')
    ax.set(xlabel="t [s]", ylabel='epsilon [m]')
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


def plot_distance_to_obs(__stored_data, __end_idx, start_idx = 0):
    # # PLOT THE DETECTION
    # # ---------------------------------------------------
    fig = plt.figure(figsize=(6.4, 2.0))
    plt.rcParams.update({'font.size': FS})
    plt.rcParams['text.usetex'] = True
    ax = plt.gca()
    # Swap variables to get start index at 1
    __stored_data['lidar_4'] = __stored_data['lidar_3']
    __stored_data['lidar_3'] = __stored_data['lidar_2']
    __stored_data['lidar_2'] = __stored_data['lidar_1']
    __stored_data['lidar_1'] = __stored_data['lidar_0']
    del __stored_data['lidar_0']
    plot_pickle_log_time_series_batch_keys(ax, __stored_data, __end_idx, 'lidar_', start_idx=start_idx)
    # plot_pickle_log_time_series_batch_keys(ax, __stored_data, __end_idx, 'h_lidar_')
    ax.plot(__stored_data['time'][:__end_idx], [SceneSetup.d_obs for _ in range(__end_idx)], linestyle='dashed', color='k', label='$R_s$')
    ax.set(xlabel="t [s]", ylabel='$\min_k ||p_i - p_{i,k}^{\mathrm{obs}}||$')
    ax.legend(loc='lower right', prop={'size': leg_size}, ncol=3, fancybox=True, shadow=True)
    # plt.show()
    figname = SimSetup.sim_defname + '_dist_to_obs.pdf'
    pngname = SimSetup.sim_defname + '_dist_to_obs.png'
    plt.savefig(figname, bbox_inches="tight", dpi=300)
    plt.savefig(pngname, bbox_inches="tight", dpi=300)
    print('export figure: ' + pngname, flush=True)


def plot_distance_formation(__stored_data, __end_idx):
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
    # fig, ax = plt.subplots(nrows=5, ncols=1, sharex=True, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 1, 4, 4, 4]})
    fig, ax = plt.subplots(nrows=5, ncols=1, sharex=True, figsize=(6.4, 9.6), gridspec_kw={'height_ratios': [1, 1, 4, 4, 4]})
    plt.rcParams.update({'font.size': FS})

    # Subplot 1
    plot_pickle_log_time_series_batch_keys(ax[0], __stored_data, __end_idx, 'dist_0_1')
    ax[0].fill_between(__stored_data['time'][:__end_idx], __stored_data['disl_0_1'], __stored_data['disu_0_1'],
                        alpha=0.2, color='k', linewidth=0)
    ax[0].plot(__stored_data['time'][:__end_idx], __stored_data['disD_0_1'], color='k', linestyle='dashed', linewidth=1)
    ax[0].plot(__stored_data['time'][:__end_idx], __stored_data['disU_0_1'], color='r', linestyle='dashed', linewidth=1)
    ax[0].plot(__stored_data['time'][:__end_idx], __stored_data['disL_0_1'], color='r', linestyle='dashed', linewidth=1)
    margin = 0.2*(__stored_data['disU_0_1'][0] - __stored_data['disL_0_1'][0])
    ax[0].set(ylim=(__stored_data['disL_0_1'][0] - margin, __stored_data['disU_0_1'][0] + margin))

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
    margin = 0.2*(__stored_data['disU_2_3'][0] - __stored_data['disL_2_3'][0])
    ax[1].set(ylim=(__stored_data['disL_2_3'][0] - margin, __stored_data['disU_2_3'][0] + margin))

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
    margin = 0.2*(__stored_data['disU_1_2'][0] - __stored_data['disL_1_2'][0])    
    ax[2].set(ylim=(__stored_data['disL_1_2'][0] - margin, __stored_data['disU_1_2'][0] + margin))

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
    margin = 0.2*(__stored_data['disU_0_3'][0] - __stored_data['disL_0_3'][0])    
    ax[3].set(ylim=(__stored_data['disL_0_3'][0] - margin, __stored_data['disU_0_3'][0] + margin))

    # Subplot 5
    l0 = plot_pickle_log_time_series_batch_keys(ax[4], __stored_data, __end_idx, 'dist_0_2')
    l1 = ax[4].fill_between(__stored_data['time'][:__end_idx], __stored_data['disl_0_2'], __stored_data['disu_0_2'],
                        alpha=0.2, color='k', linewidth=0)
    l2, = ax[4].plot(__stored_data['time'][:__end_idx], __stored_data['disD_0_2'], color='k', linestyle='dashed',
                linewidth=1)
    l3, = ax[4].plot(__stored_data['time'][:__end_idx], __stored_data['disU_0_2'], color='r', linestyle='dashed',
                linewidth=1)
    ax[4].plot(__stored_data['time'][:__end_idx], __stored_data['disL_0_2'], color='r', linestyle='dashed',
                linewidth=1)
    margin = 0.2*(__stored_data['disU_0_2'][0] - __stored_data['disL_0_2'][0])    
    ax[4].set(ylim=(__stored_data['disL_0_2'][0] - margin, __stored_data['disU_0_2'][0] + margin))

    # Axes title
    ax[0].set_ylabel("$\Vert p_1-p_2 \Vert$", rotation=0, labelpad=40)
    ax[1].set_ylabel("$\Vert p_3-p_4 \Vert$", rotation=0, labelpad=40)
    ax[2].set_ylabel("$\Vert p_2-p_3 \Vert$", rotation=0, labelpad=40)
    ax[3].set_ylabel("$\Vert p_1-p_4 \Vert$", rotation=0, labelpad=40)
    ax[4].set_ylabel("$\Vert p_1-p_3 \Vert$", rotation=0, labelpad=40)
    ax[4].set_xlabel("t [s]")

    hdl = [l0, l2, l1, l3]
    lbl = [
        '$\Vert p_i-p_j \Vert$', 
        '$d_{ij}^{\mathrm{desired}}$', 
        '$\pm (\epsilon_i + \epsilon_j)$',
        '$\pm \epsilon_{ij}^{\mathrm{max}}$'
        ]
    ax[4].legend(handles = hdl , labels=lbl, loc='upper center', 
             bbox_to_anchor=(0.35, -0.3),fancybox=True, shadow=True, ncol=4)

    # Save
    figname = SimSetup.sim_defname + '_form_dist.pdf'
    pngname = SimSetup.sim_defname + '_form_dist.png'
    plt.savefig(figname, bbox_inches="tight", dpi=300)
    plt.savefig(pngname, bbox_inches='tight',dpi=300)
    print('export figure: ' + pngname, flush=True)



# ---------------------------------------------------
# GENERIC PLOTTING FUNCTIONS
# ---------------------------------------------------
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

def plot_pickle_log_time_series_batch_keys(ax, datalog_data, __end_idx, pre_string, start_idx=0):
    """
    :param ax:
    :param datalog_data:
    :param __end_idx:
    :param pre_string:
    :return:
    """
    # check all matching keystring
    time_data = datalog_data['time'][start_idx:__end_idx]
    matches = [key for key in datalog_data if key.startswith(pre_string)]
    data_min, data_max = 0., 0.
    for key in matches:
        key_data = datalog_data[key][start_idx:__end_idx]
        # key_data = key_data.reshape(key_data.shape[0], -1) if len(key_data.shape) > 2 else key_data
        l, = ax.plot(time_data, key_data, label=key.strip(pre_string) if len(key.strip(pre_string)) > 0 else pre_string)
        # update min max for plotting
        for i in key_data:
            if i is not None and i is not np.nan:
                data_min, data_max = min(data_min, i), max(data_max, i)
    ax.grid(True)
    ax.set(xlim=(time_data[0] - 0.1, time_data[-1] + 0.1),
           ylim=(data_min - 0.1, data_max + 0.1))
    
    return l

