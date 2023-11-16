from nebolab_experiment_setup import NebolabSetup
from scenarios_unicycle.CCTA2024_Controller import SceneSetup, np

import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.gridspec import GridSpec
from simulator.dynamics import Unicycle
from simulator.plot_2D_unicycle import draw2DUnicyle
from simulator.data_logger import dataLogger
from simulator.detect_obstacle import DetectObstacle


# TODO: setup PEP for the variable naming, formatting, etc.

# GENERAL PARAM AND COMPUTATION FOR THIS SPECIFIC SCENARIO (Both SIM and EXP)
# -----------------------------------------------------------------------

def import_scenario():
    with open('scenarios_unicycle/scenarios/formation4_exp.yml', 'r') as file:
        import yaml
        scenario, control, setup = yaml.safe_load(file).values()

    # Set the static obstacles
    SceneSetup.static_obstacles = [np.array(o) for o in scenario['obstacles']]

    # CONTROL PARAMETER
    [exec(f'SceneSetup.{k} = {v}') for data in control.values() for k, v in data.items()]
    SceneSetup.robot_num = len(scenario['formations']['structs'])
    # TODO: this should not be like this
    SceneSetup.robot_color = [f'{np.random.random():.3f}' for _ in range(SceneSetup.robot_num)]
    SceneSetup.default_range_data = np.ones((SceneSetup.robot_num,
                                             SceneSetup.sensor_resolution)) * SceneSetup.default_range

    # Formation setup
    from scipy.linalg import block_diag
    # SceneSetup.hull = block_diag(*[np.array(h) for h in scenario['formations']['hulls']])
    form_l = scenario['formations']['form_scaling']
    SceneSetup.form_num = len(scenario['formations']['links'])
    SceneSetup.form_size = [len(form) for form in scenario['formations']['links']]
    SceneSetup.form_id = np.array([idx for idx in range(SceneSetup.form_num)
                                   for _ in range(SceneSetup.form_size[idx])])  # Identifier for each group
    SceneSetup.max_form_epsilon = block_diag(*[np.array(mfe) for mfe in scenario['formations']['max_form_epsilon']])

    # Define the leader ID in each formation and the initial offset to major axis
    SceneSetup.struct = np.array([np.array(scenario['formations']['structs'][idx]) * form_l[SceneSetup.form_id[idx]]
                                  for idx in range(SceneSetup.robot_num)])
    # SceneSetup.form_A = block_diag(*[np.array(f) for f in scenario['formations']['links']])
    SceneSetup.form_A = block_diag(*[np.array(scenario['formations']['links'][idx]) * form_l[idx]
                                     for idx in range(SceneSetup.form_num)])
    SceneSetup.form_leader_id = np.array([SceneSetup.form_id.tolist().index(idx)
                                          for idx in range(SceneSetup.form_num)])
    SceneSetup.form_leader_offset = np.array([np.arctan2(SceneSetup.struct[idx][1], SceneSetup.struct[idx][0])
                                              for idx in SceneSetup.form_leader_id])  # np.array([0., 0.])

    # print(SceneSetup.struct)
    # SceneSetup.form_A_eps = np.where(SceneSetup.form_A > 0, SceneSetup.initial_eps * SceneSetup.max_form_epsilon, 0)
    SceneSetup.form_A_eps = SceneSetup.initial_eps * SceneSetup.max_form_epsilon
    SceneSetup.form_A_edges = np.count_nonzero(SceneSetup.form_A_eps, axis=1)

    # Working with Waypoints
    SceneSetup.form_waypoints = {idx: np.array(scenario['positions']['waypoints'][idx])[:, :-1]
                                 for idx in range(SceneSetup.form_num)}
    SceneSetup.form_wp_orient = {idx: np.array(scenario['positions']['waypoints'][idx])[:, -1]
                                 for idx in range(SceneSetup.form_num)}
    SceneSetup.wp_switch_radius = scenario['positions']['wp_switch_radius']

    # Set initial formation position
    rot = lambda alpha: np.array([[np.cos(alpha), -np.sin(alpha), 0.],
                                  [np.sin(alpha), np.cos(alpha), 0.],
                                  [0., 0., 1.]])
    SceneSetup.init_pos = np.array([
        scenario['positions']['initial_positions'][SceneSetup.form_id[idx]][:-1]
        + rot(scenario['positions']['initial_positions'][SceneSetup.form_id[idx]][-1]) @ SceneSetup.struct[idx]
        for idx in range(SceneSetup.robot_num)
    ])
    SceneSetup.init_theta = np.array([0 for _ in range(SceneSetup.robot_num)])

    SceneSetup.goal_pos = np.array([
        SceneSetup.form_waypoints[SceneSetup.form_id[idx]][0]
        + rot(SceneSetup.form_wp_orient[SceneSetup.form_id[idx]][0]) @ SceneSetup.struct[idx]
        for idx in range(SceneSetup.robot_num)
    ])

    SceneSetup.major_l = scenario['formations']['major_l']  # diameter of ellipse in major-axis
    SceneSetup.minor_l = scenario['formations']['minor_l']  # ... in minor-axis
    NebolabSetup.FIELD_X = SceneSetup.FIELD_X
    NebolabSetup.FIELD_Y = SceneSetup.FIELD_Y

    print('robot_num\n', SceneSetup.robot_num)
    # print('default_range_data\n', SceneSetup.default_range_data)
    print('form_wp\n', SceneSetup.form_waypoints)
    print('form_wp_orient\n', SceneSetup.form_wp_orient)
    print('init_pos\n', SceneSetup.init_pos)
    print('goal_pos\n', SceneSetup.goal_pos)
    print('init_theta\n', SceneSetup.init_theta)
    print('major_l\n', SceneSetup.major_l)
    print('minor_l\n', SceneSetup.minor_l)
    print('wp_switch_radius\n', SceneSetup.wp_switch_radius)
    print('struct\n', SceneSetup.struct)
    print('form_A\n', SceneSetup.form_A)
    print('form_A_edges\n', SceneSetup.form_A_edges)
    print('form_leader_id\n', SceneSetup.form_leader_id)
    print('form_leader_offset\n', SceneSetup.form_leader_offset)
    print('form_size\n', SceneSetup.form_size)
    print('form_id\n', SceneSetup.form_id)
    print('max_form_epsilon\n', SceneSetup.max_form_epsilon)
    print('d_obs\n', SceneSetup.d_obs)

    # Modify SimSetup
    [exec(f'SimSetup.{k} = {v}') for k, v in setup.items()]
    SimSetup.robot_angle_bound = np.pi / SimSetup.bound_vertices + \
                                 np.append(np.linspace(0., 2 * np.pi, num=SimSetup.bound_vertices, endpoint=False), 0)


# ONLY USED IN SIMULATION
# -----------------------------------------------------------------------
class SimSetup:
    Ts = 0.02  # in second. Determine Visualization and dynamic update speed
    tmax = 60  # simulation duration in seconds (only works when save_animate = True)
    save_animate = False  # True: saving but not showing, False: showing animation but not real time
    save_data = False  # log data using pickle
    plot_saved_data = False

    trajectory_trail_lenTime = tmax  # in second, to show all trajectory use tmax

    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")
    # desc = "testVis"
    desc = "exp"
    sim_defname = f'animation_result/{dt_string}_{desc}/sim2D_FormationObstacleLidar'
    sim_fname_output = r'' + sim_defname + '.gif'
    sim_trajTail = None  # Show all trajectory
    sim_fdata_vis = sim_defname + '_vis.pkl'

    timeseries_window = 1  # in seconds, for the time series data
    eps_visualization = True
    DETECT_OTHER_ROBOTS = True
    robot_angle_bound = np.append(np.linspace(0., 2 * np.pi, num=8, endpoint=False), 0) + np.pi / 8
    robot_rad = 0.1


# General class for drawing the plots in simulation
class SimulationCanvas:
    def __init__(self):
        # NOTE: ALWAYS DO THIS FIRST
        import_scenario()
        self.__cur_time = 0.

        # Initiate the robot's dynamics
        self.__robot_dyn = [None for _ in range(SceneSetup.robot_num)]
        for i in range(SceneSetup.robot_num):
            self.__robot_dyn[i] = Unicycle(SimSetup.Ts, SceneSetup.init_pos[i], ell=NebolabSetup.TB_L_SI2UNI)

        # Initiate data_logger
        self.log = dataLogger(round(SimSetup.tmax / SimSetup.Ts) + 1)

        # Initiate ranging sensors for the obstacles
        self.__rangesens = DetectObstacle(detect_max_dist=SceneSetup.default_range,
                                          angle_res_rad=2 * np.pi / SceneSetup.sensor_resolution)
        for i in range(len(SceneSetup.static_obstacles)):
            self.__rangesens.register_obstacle_bounded('obs' + str(i), SceneSetup.static_obstacles[i])

        # Display sensing data
        self.__pl_sens = dict()

        # Initiate the plotting
        self.__initiate_plot()
        self.log.save_description(SimSetup.sim_fdata_vis, SceneSetup)

        # flag to check if simulation is still running
        self.is_running = True

    def update_simulation(self, control_input, feedback):
        if self.__cur_time < SimSetup.tmax:
            # Store data to log
            self.log.store_dictionary(control_input.get_all_monitored_info())
            self.log.time_stamp(self.__cur_time)

            self.__cur_time += SimSetup.Ts
            # Set array to be filled
            all_robots_pos = np.zeros(SceneSetup.init_pos.shape)
            all_robots_theta = np.zeros(SceneSetup.init_theta.shape)
            all_range_data = SceneSetup.default_range_data.copy()

            # IMPORTANT: advance the robot's dynamic, and update feedback information
            for i in range(SceneSetup.robot_num):
                self.__robot_dyn[i].set_input(control_input.get_i_vel_xy(i), "u")
                state = self.__robot_dyn[i].step_dynamics()

                all_robots_pos[i, :2] = state['q'][:2]
                all_robots_theta[i] = state['theta']

            if self.__cur_time % SimSetup.LiDAR_rate < SimSetup.Ts:
                print(self.__cur_time)
                for i in range(SceneSetup.robot_num):
                    if SimSetup.DETECT_OTHER_ROBOTS:
                        # Update robot shape to be used for range detection
                        v_angles = SimSetup.robot_angle_bound + all_robots_theta[i]
                        robot_shape = np.array([np.cos(v_angles), np.sin(v_angles), v_angles * 0]) * SimSetup.robot_rad
                        robot_bounds = np.transpose(robot_shape + all_robots_pos[i, :3].reshape(3, 1))
                        self.__rangesens.register_obstacle_bounded(i, robot_bounds)

                for i in range(SceneSetup.robot_num):
                    # update sensor data by excluding its own
                    all_range_data[i, :] = self.__rangesens.get_sensing_data(
                        all_robots_pos[i, 0], all_robots_pos[i, 1], all_robots_theta[i],
                        exclude=[i] if SimSetup.DETECT_OTHER_ROBOTS else [])

                feedback.set_sensor_reading(all_range_data)

            # UPDATE FEEDBACK for the controller
            feedback.set_feedback(all_robots_pos, all_robots_theta)


            #TODO: put this in expenv
            if SimSetup.eps_visualization:  # TODO WIDHI: check notes in SimSetup
                eps_array = control_input.get_all_epsilons()
                feedback.set_all_eps(eps_array)
                # print("Epsilons:", eps_array.T[0])

        else:  # No further update
            if self.is_running:
                if SimSetup.save_data:
                    self.log.save_to_pkl(SimSetup.sim_fdata_vis)
                    # TODO: plot data further from saved pickle
                    if SimSetup.plot_saved_data:
                        from pickleplot import scenario_pkl_plot
                        scenario_pkl_plot()
                print(f"Stopping the simulation, tmax reached: {self.__cur_time:.2f} s")
                # if not SimSetup.save_animate: exit() # force exit
                self.is_running = False
                # else: # Do nothing

        # Update plot
        self.__update_plot(feedback)

    # PROCEDURES RELATED TO PLOTTING - depending on the scenarios
    # ---------------------------------------------------------------------------------
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
                                                                                 color='k', linewidth=0.5 / SceneSetup.max_form_epsilon[i][j])

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


# ONLY USED IN EXPERIMENT
# -----------------------------------------------------------------------
class ExpSetup():
    exp_defname = 'experiment_result/ROSTB_FormationAvoidance'
    exp_fdata_vis = exp_defname + '_data.pkl'
    ROS_RATE = 50


class ExperimentEnv():
    def __init__(self):
        # NOTE: ALWAYS DO THIS FIRST
        import_scenario()
        self.global_lahead = [None for _ in range(SceneSetup.robot_num)]
        self.global_poses = [None for _ in range(SceneSetup.robot_num)]
        # self.scan_LIDAR = [None for _ in range(SceneSetup.robot_num)]
        self.scan_LIDAR = SceneSetup.default_range_data.copy()
        # Initiate data_logger
        self.__cur_time = 0.
        self.log = dataLogger( ExpSetup.log_duration * ExpSetup.ROS_RATE )

    # NOTES: it seems cleaner to do it this way
    # rather than dynamically creating the callbacks
    def pos_callback(self, msg, index): self.global_lahead[index] = msg
    def posc_callback(self, msg, index): self.global_poses[index] = msg
    def scan_LIDAR_callback(self, msg, index): self.scan_LIDAR[index, :] = np.array(msg.ranges)

    def update_feedback(self, feedback, control_input):
        all_robots_pos = np.zeros([SceneSetup.robot_num, 3])
        all_robots_theta = np.zeros([SceneSetup.robot_num, 1])
        # TODO: below might not work for robots less than 4
        for i in range(SceneSetup.robot_num):
            all_robots_pos[i,0] = self.global_poses[i].x
            all_robots_pos[i,1] = self.global_poses[i].y
            all_robots_theta[i] = self.global_poses[i].theta
        # UPDATE FEEDBACK for the controller
        feedback.set_feedback(all_robots_pos, all_robots_theta)

        eps_array = control_input.get_all_epsilons()
        feedback.set_all_eps(eps_array)
        feedback.set_sensor_reading(self.scan_LIDAR)

    def get_i_vlin_omega(self, i, control_input):
        # Inverse Look up ahead Mapping (u_z remain 0.)
        #   V = u_x cos(theta) + u_y sin(theta)
        #   omg = (- u_x sin(theta) + u_y cos(theta)) / l
        u = control_input.get_i_vel_xy(i)
        theta = self.global_poses[i].theta
        vel_lin = u[0]*np.cos(theta) + u[1]*np.sin(theta)
        vel_ang = (- u[0]*np.sin(theta) + u[1]*np.cos(theta))/NebolabSetup.TB_L_SI2UNI
        return vel_lin, vel_ang

    def update_log(self, control_input):
        # Store data to log
        self.log.store_dictionary( control_input.get_all_monitored_info() )
        self.log.time_stamp( self.__cur_time )
        # NOT REAL TIME FOR NOW. TODO: fix this with real time if possible
        self.__cur_time += 1./ExpSetup.ROS_RATE

    def save_log_data(self):
        self.log.save_to_pkl( ExpSetup.exp_fdata_vis )
        # automatic plot if desired
        # if SimSetup.plot_saved_data:
        #     from scenarios.Resilient_pickleplot import scenario_pkl_plot
        #     # Quick fix for now --> TODO: fix this
        #     SimSetup.sim_defname = ExpSetup.exp_defname
        #     SimSetup.sim_fdata_log = ExpSetup.exp_fdata_log
        #     # Plot the pickled data
        #     scenario_pkl_plot()
