from control_lib.go_to_goal import Pcontrol, waypoint_planner
from control_lib.cbf_single_integrator import cbf_si, np
from nebolab_experiment_setup import NebolabSetup


# MAIN COMPUTATION FOR CCTA2022 - This code should be general for all scenarios
# ------------------------------------------------------------------------------
class SceneSetup:
    """
    General variable needed to run the controller
    Can be adjusted later by set new value on the class variable
    """
    robot_num: int = 4
    # Set initial formation position --> Order: red, blue, green, orange
    init_pos: np.array = np.array([[-1, 1.25, 0],
                                   [-1, 0.75, 0],
                                   [-1.02, -0.75, 0],
                                   [-1.02, -1.25, 0]])
    init_theta: np.array = np.array([0, 0, 0, 0])
    # Set desired formation position --> rotated -90deg in final configuration
    goal_pos: np.array = np.array([[1.25, -1, 0],
                                   [0.75, -1, 0],
                                   [0.75, 1, 0],
                                   [1.25, 1, 0]])
    # Here we assume any final pose is OK
    robot_color: [] = []

    # OBSTACLE PARAMETER
    static_circle_obstacles: [] = [{"pos": np.array([-0.4, 0.6, 0]), "r": 0.3},
                                   {"pos": np.array([-0.5, -1., 0]), "r": 0.3}]
    static_obstacles: [] = []

    # FORMATION PARAMETER
    # distance to keep between each robot
    form_A: np.array = np.array([[0, 0.5, 0, 0],
                                 [0.5, 0, 0, 0],
                                 [0, 0, 0, 0.5],
                                 [0, 0, 0.5, 0]])
    form_A_eps: np.array = form_A.copy()  # adjacent matrix for epsilon
    form_A_edges: np.array = np.array([1, 1, 1, 1])  # number of edges on each vertex
    max_form_epsilon: float = 0.2  # tolerance for maintaining distance in form_A
    hull: np.array = None

    form_num: int = 2
    form_id: np.array = np.array([0, 0, 1, 1])  # Identifier for each group
    # Define the leader ID in each formation and the initial offset to major axis
    form_leader_id = np.array([0, 2])
    form_leader_offset = np.array([0., 0.])
    major_l = [2 * 0.6 for _ in range(form_num)]  # diameter of ellipse in major-axis
    minor_l = [2 * 0.4 for _ in range(form_num)]  # ... in minor-axis

    # CONTROL PARAMETER
    USECBF_BOUNDARY = False  # Control input boundary
    USECBF_STATICOBS = False  # Static circular obstacle
    USECBF_FORMATION = False  # Formation Maintenance
    USECBF_LIDAR = False  # LiDAR option
    USECBF_LIDAR_SHARING = False

    use_unicycle = True

    Pgain: float = 0.8  # for go-to-goal
    eps_gain: float = 1.  # for achieving eps → 0
    eps_weight: float = 1.  # priority in optimizing function
    initial_eps: float = 0.5  # ratio
    gamma_staticObs: float = 10
    gamma_form: float = 10
    default_range_data: np.array = np.zeros((0, 0))
    sensor_resolution: int = 360  # LiDAR resolution, shall not change
    default_range: float = 1.  # LiDAR sensor range
    kappa: float = 0.16  # relative active obstacles threshold
    d_obs: float = 0.4  # minimum distance to obstacle, LiDAR case
    robot_offset: float = 0.

    speed_limit: float = 0.

    # define sensing range (to ease optimization)
    sr = 10  # in meter # It needs to be greater than major length
    ds_dyn = 0.1


class Controller:
    """
    General class for computing the controller input
    """

    def __init__(self):
        """
        Initialize Controller
        """
        # Re-estimate in_neighbor, for now assume formation graph is identical with communication graph
        self.__inNeigh = [np.where(SceneSetup.form_A[:, i] > 0)[0] for i in range(SceneSetup.robot_num)]
        # Setup cbf
        self.cbf = [cbf_si(neighbor_eps=SceneSetup.form_A_eps[i]) for i in range(SceneSetup.robot_num)]

        if SceneSetup.USE_WAYPOINTS:
            # Setup Waypoint manager
            self.wp_manager = {}
            for f_id in range(SceneSetup.form_num):
                wp = SceneSetup.form_waypoints[f_id]
                orient = SceneSetup.form_wp_orient[f_id]
                self.wp_manager[f_id] = waypoint_planner(wp, orient)

    def compute_control(self, feedback, computed_control):
        """
        Apply different controller for each scenario

        :param feedback: feedback class, contains all the required information for the controller computation
        :param computed_control: is also the output
        """
        # Reset monitor properties
        computed_control.reset_monitor()

        if SceneSetup.USE_WAYPOINTS:
            # Compute centroid centralized (for now. TODO: distribute later)
            rot = lambda alpha: np.array([[np.cos(alpha), -np.sin(alpha), 0.],
                                          [np.sin(alpha), np.cos(alpha), 0.],
                                          [0., 0., 1.]])

            for f_id in range(SceneSetup.form_num):
                form_pos, form_th = feedback.get_form_i_state(f_id)  # get centroid
                # Recheck waypoint
                is_update, new_wp, new_orient = \
                    self.wp_manager[f_id].check_wp_status(form_pos, SceneSetup.wp_switch_radius)
                if is_update:  # update new goal position for each robot
                    # SceneSetup.form_waypoints.pop(0)
                    for idx in range(SceneSetup.robot_num):
                        if f_id == SceneSetup.form_id[idx]:
                            SceneSetup.goal_pos[idx, :] = new_wp + rot(new_orient) @ SceneSetup.struct[idx]

        for i in range(SceneSetup.robot_num):
            # Collect States
            # ------------------------------------------------
            current_q = feedback.get_robot_i_pos(i)  # Get position data only

            if SceneSetup.use_unicycle:
                current_q = feedback.get_lahead_i_pos(i)
            goal = SceneSetup.goal_pos[i]

            # Implementation of Control
            # ------------------------------------------------
            # Calculate nominal controller
            u_nom = Pcontrol(current_q, SceneSetup.Pgain, goal)
            i_eps = feedback.get_robot_i_eps(i)

            # Store basic data to monitor
            computed_control.save_monitored_info(f"pos_x_{i}", current_q[0])
            computed_control.save_monitored_info(f"pos_y_{i}", current_q[1])
            computed_control.save_monitored_info(f"goal_x_{i}", goal[0])
            computed_control.save_monitored_info(f"goal_y_{i}", goal[1])

            # Construct CBF setup
            self.cbf[i].reset_cbf()

            # Static Circular Robot-Obstacle Avoidance
            if SceneSetup.USECBF_STATICOBS:
                for idx, obs in enumerate(SceneSetup.static_circle_obstacles):
                    h = self.cbf[i].add_avoid_static_circle(current_q, obs["pos"], obs["r"],
                                                            gamma=SceneSetup.gamma_staticObs, power=3)
                    # store h value
                    computed_control.save_monitored_info(f"h_staticobs_{i}_{idx}", h)

            # Formation Maintenance
            if SceneSetup.USECBF_FORMATION:
                for j in self.__inNeigh[i]:
                    j_q = feedback.get_robot_i_pos(j)
                    j_eps = feedback.get_robot_i_eps(j)
                    if SceneSetup.use_unicycle:
                        j_q = feedback.get_lahead_i_pos(j)

                    h_fml, h_fmu, h_eps_floor, h_eps_ceil = self.cbf[i].add_maintain_distance_with_distinct_epsilon(
                        current_q, j_q, SceneSetup.form_A[i, j], SceneSetup.max_form_epsilon[i][j], i_eps[j], j_eps[i],
                        j,
                        gamma=10, power=3)

                    # store h value
                    computed_control.save_monitored_info(f"h_eps_ceil_{i}_{j}", h_eps_ceil)
                    computed_control.save_monitored_info(f"h_eps_floor_{i}_{j}", h_eps_floor)
                    computed_control.save_monitored_info(f"h_fml_{i}_{j}", h_fml)
                    computed_control.save_monitored_info(f"h_fmu_{i}_{j}", h_fmu)

            # Non-circular Robot-Obstacle Avoidance
            if SceneSetup.USECBF_LIDAR:
                # Get the LIDAR data
                range_data = feedback.get_robot_i_range_data(i)  # [(0 → 1)]
                range_points = feedback.get_robot_i_detected_pos(i)  # [(obs_x, obs_y, 0)]
                detected_obs_points = range_points[range_data < 0.99 * SceneSetup.default_range]  # filtered

                min_h = self.cbf[i].add_avoid_lidar_detected_obs(
                    detected_obs_points, current_q,
                    SceneSetup.kappa, SceneSetup.d_obs,
                    gamma=SceneSetup.gamma_staticObs
                )

                # store h value
                computed_control.save_monitored_info(f"h_staticobs_{i}", min_h)

            if SceneSetup.speed_limit > 0.:
                # set speed limit
                norm = np.hypot(u_nom[0], u_nom[1])

                if norm > SceneSetup.speed_limit:
                    u_nom = SceneSetup.speed_limit * u_nom / norm  # max
                # self.cbf[i].add_velocity_bound(SceneSetup.speed_limit)

            # Ensure safety
            # print(f'Robot {i}')
            u, v = self.cbf[i].compute_safe_controller(u_nom, -SceneSetup.eps_gain * i_eps,
                                                       weight=SceneSetup.eps_weight)

            # Store command
            # ------------------------------------------------
            computed_control.set_i_vel_xy(i, u[:2])

            self.cbf[i].update_additional_state((Ts := 0.02))
            i_eps = self.cbf[i].get_additional_state()
            for idx, eps in enumerate(i_eps.tolist()):
                computed_control.save_monitored_info(f"eps_{i}_{idx}", eps)
            computed_control.set_i_eps(i, i_eps)

            # Save u_nom and u* to pkl
            computed_control.save_monitored_info(f"u_nom_x_{i}", u_nom[0])
            computed_control.save_monitored_info(f"u_nom_y_{i}", u_nom[1])
            computed_control.save_monitored_info(f"u_x_{i}", u[0])
            computed_control.save_monitored_info(f"u_y_{i}", u[1])


class ControlOutput:
    """
    CLASS FOR CONTROLLER'S INPUT AND OUTPUT
    Encapsulate the control input passing from controller into sim/experiment
    """

    def __init__(self):
        """
        Initialize the formation array
        """
        self.__all_velocity_input_xyz = np.zeros([SceneSetup.robot_num, 3])
        self.__all_eps = SceneSetup.form_A_eps.copy()
        self.reset_monitor()

    def get_all_vel_xy(self):
        """
        __all_velocity_input_xyz getter
        :return: Nx2 vector, velocity on each axis of each robot
        """
        return self.__all_velocity_input_xyz[:, :2]

    def get_all_epsilons(self):
        """
        __all_eps getter
        :return: Nx1 vector, epsilon of each robot
        """
        # print(self.__all_eps)
        return self.__all_eps.copy()

    def get_i_epsilon(self, ID):
        """
        __all_eps getter
        :return: Nx1 vector, epsilon of each robot
        """
        return self.__all_eps[ID, :]

    def get_i_vel_xy(self, ID):
        """
        __all_velocity_input_xyz @ row ID-th getter

        :param ID: ID-th robot in the simulation
        :return: 1x2 vector, velocity of ID-th robot
        """
        return self.__all_velocity_input_xyz[ID, :2]

    def set_i_vel_xy(self, ID, input_xy):
        """
        __all_velocity_input_xyz @ row ID-th setter

        :param ID: integer, ID-th robot in the simulation
        :param input_xy: 1x2 vector, new velocity
        """
        self.__all_velocity_input_xyz[ID, :2] = input_xy

    def set_i_eps(self, ID, input_eps):
        """
        __all_eps @ row ID-th setter

        :param ID: integer, ID-th robot in the simulation
        :param input_eps: scalar, new epsilon
        """
        self.__all_eps[ID, :] = input_eps

    # Allow the options to monitor state / variables over time
    def reset_monitor(self):
        """
        __monitored_signal setter
        Clear the communication dictionary
        """
        self.__monitored_signal = {}

    def save_monitored_info(self, label, value):
        """
        __monitored_signal setter at specific key
        By default name the label with the index being the last
        i.e. p_x_0, p_y_0, h_form_1_2, etc.

        :param label: string value, name of signal
        :param value: scalar value, value of signal
        """

        self.__monitored_signal[label] = value

    # Allow retrieval from sim or experiment
    def get_all_monitored_info(self):
        """
        __monitored_signal getter
        Signal retrieval

        :return: dictionary, key-value as name and value of signal
        """
        return self.__monitored_signal


class FeedbackInformation:
    """
    Encapsulate the feedback passing from sim/experiment into controller
    """

    def __init__(self):
        """
        Initialize the formation array, POI, feedback
        """
        # Initialize the formation array
        self.__all_form_centroid = np.zeros([SceneSetup.robot_num, 3])
        self.__all_form_theta = np.zeros([SceneSetup.robot_num, 1])
        # Initialize lookahead position for all robots
        self.__all_lahead_pos = np.zeros([SceneSetup.robot_num, 3])
        self.__all_robot_epsilon = SceneSetup.form_A_eps.copy()

        # Set the value based on initial values
        self.set_feedback(SceneSetup.init_pos, SceneSetup.init_theta)

        if SceneSetup.USECBF_LIDAR:
            # Set the range data and detected pos
            n, m = SceneSetup.default_range_data.shape
            self.__all_detected_pos = np.zeros((n, m, 3))
            self.__sensing_linspace = np.linspace(0., 2 * np.pi, num=m, endpoint=False)
            self.set_sensor_reading(SceneSetup.default_range_data)

        self.__all_robot_epsilon_hist = self.__all_robot_epsilon.copy().reshape(SceneSetup.form_A_eps.shape[0],
                                                                                SceneSetup.form_A_eps.shape[1], 1)

    @staticmethod
    def __compute_ellipse_formation(form_robots_pos, leader_pos, leader_offset_rad):
        """
        Simulate getting other group's computation via communication

        :param form_robots_pos: Nx2 vector, position of all robots in formation
        :param leader_pos: 2x1 vector, position of communication node
        :param leader_offset_rad: scalar value, fixed angle between
                communication node and center of formation
        :return cent: 2x1 vector, center of formation
        :return theta: scalar value, orientation of formation
        """
        # Compute ellipse centroids
        cent = np.sum(form_robots_pos, 0) / form_robots_pos.shape[0]
        # Compute vector for each ellipse's theta
        vector = leader_pos - cent  # robot 0 is leader group 1
        theta = np.arctan2(vector[1], vector[0]) + leader_offset_rad  # no offset for now
        return cent, theta

    def set_feedback(self, all_robots_pos, all_robots_theta, all_robots_eps=None):
        """
        Update position, orientation, POI of all robots and formations

        :param all_robots_pos: Nx2 matrix, position of all robots
        :param all_robots_theta: Nx1 vector, orientation of all robots
        :param all_robots_eps: NxN matrix, epsilon of all links between robots
        """
        # update all robots position, theta, and epsilon (if not omitted)
        self.__all_robot_pos = all_robots_pos.copy()
        self.__all_robot_theta = all_robots_theta.copy()
        if all_robots_eps is not None:
            self.__all_robot_epsilon = all_robots_eps.copy()
        # update lookahead position for each robot
        for i in range(SceneSetup.robot_num):
            th = all_robots_theta[i]
            ell_si = np.array([np.cos(th), np.sin(th), 0], dtype=object) * NebolabSetup.TB_L_SI2UNI
            self.__all_lahead_pos[i, :] = all_robots_pos[i, :] + ell_si
        # update ellipse formation
        for i_form in range(SceneSetup.form_num):
            form_robots_pos = self.__all_lahead_pos[np.where(SceneSetup.form_id == i_form)[0], :]
            leader_pos = self.__all_lahead_pos[SceneSetup.form_leader_id[i_form], :]
            self.__all_form_centroid[i_form, :], self.__all_form_theta[i_form, :] = \
                self.__compute_ellipse_formation(form_robots_pos, leader_pos, SceneSetup.form_leader_offset[i_form])

        # TODO: use the SceneSetup.formA to make the above adjustable
        # NOTE: I think we can set the compute_ellipse_form2 to directly check for all existing id

    def set_sensor_reading(self, all_range_data):
        """
        __all_detected_pos setter
        :param all_range_data: NxMx2 matrix, sensing data package of N robots with M resolution
        """
        self.__all_range_data = all_range_data.copy()
        # print('all_range_data\n', all_range_data)
        # update the detected position for each robot
        for i in range(all_range_data.shape[0]):
            # Calculate the angle of LiDAR ray in radian
            sensing_angle_rad = self.__all_robot_theta[i] + self.__sensing_linspace
            # (X, Y) of the endpoint of LiDAR ray
            self.__all_detected_pos[i, :, 0] = self.__all_robot_pos[i, 0] + all_range_data[i] * np.cos(
                sensing_angle_rad)
            self.__all_detected_pos[i, :, 1] = self.__all_robot_pos[i, 1] + all_range_data[i] * np.sin(
                sensing_angle_rad)

    # To allow access from the controller computation
    def get_robot_i_pos(self, i):
        """
        __all_robot_pos @ i-th row getter

        :param i: integer, i-th robot in simulation
        :return: 1x2 vector, position of i-th robot
        """
        return self.__all_robot_pos[i, :]

    def get_robot_i_eps(self, i):
        """
        __all_robot_epsilon @ i-th row getter

        :param i: integer, i-th robot in simulation
        :return: scalar, epsilon of i-th robot
        """
        return self.__all_robot_epsilon.copy()[i, :].flatten()

    def get_robot_i_eps_hist(self, i):
        """
        __all_robot_epsilon_hist @ i-th row getter

        :param i: integer, i-th robot in simulation
        :return: scalar, epsilon history of i-th robot
        """
        return self.__all_robot_epsilon_hist.copy()[i, :, :]

    def get_robot_i_j_eps_hist(self, i, j):
        """
        __all_robot_epsilon_hist @ i-th row getter

        :param i: integer, i-th robot in simulation
        :param j: integer, j-th robot in simulation
        :return: scalar, epsilon history of ij robots
        """
        return self.__all_robot_epsilon_hist.copy()[i, j, :]

    def get_robot_i_theta(self, i):
        """
        __all_robot_theta @ i-th row getter

        :param i: integer, i-th robot in simulation
        :return: scalar value, orientation of i-th robot
        """
        return self.__all_robot_theta[i].copy()

    def get_lahead_i_pos(self, i):
        """
        __all_lahead_pos @ i-th row getter

        :param i: integer, i-th robot in simulation
        :return: 1x2 vector, POI of i-th robot
        """
        return self.__all_lahead_pos[i, :]

    # get all robots information
    def get_all_robot_pos(self):
        """
        __all_robot_pos getter
        :return: Nx2 vector, all position of robots
        """
        return self.__all_robot_pos.copy()

    def get_all_robot_epsilon(self):
        """
        __all_robot_epsilon getter
        :return: Nx1 vector, all position of robots
        """
        return self.__all_robot_epsilon.copy()

    def get_all_robot_theta(self):
        """
        __all_robot_theta getter
        :return: Nx1 vector, all orientation of robots
        """
        return self.__all_robot_theta.copy()

    def get_all_lahead_pos(self):
        """
        __all_lahead_pos getter
        :return: Nx2 vector, all POI of robots
        """
        return self.__all_lahead_pos.copy()

    def get_form_i_state(self, i):
        """
        __all_form_centroid @ i-th row getter
        __all_form_theta @ i-th row getter

        :param i: integer, i-th robot in simulation
        :return: 1x2 vector, position of i-th robot
        :return: scalar value, orientation of i-th robot
        """
        return self.__all_form_centroid[i, :], self.__all_form_theta[i, :]

    def set_all_eps(self, epsilons):
        """
        __all_robot_epsilon setter
        :param epsilons: Nx1 vector, new epsilons
        """
        # print("v:\n", self.__all_robot_epsilon - epsilons)
        self.__all_robot_epsilon = epsilons
        self.__all_robot_epsilon_hist = np.dstack((self.__all_robot_epsilon_hist, epsilons))

    def get_all_eps(self):
        """
        __all_robot_epsilon getter
        :return: Nx1 vector, epsilons
        """
        return self.__all_robot_epsilon.copy()

    def get_robot_i_detected_pos(self, i):
        """
        __all_detected_pos @ index i-th getter

        :param i: integer, i-th robot in simulation
        :return: Mx3 matrix, m maximum-reach LiDAR scanning positions of i-th robot
        """
        return self.__all_detected_pos[i]

    def get_robot_i_range_data(self, i):
        """
        __all_range_data @ index i-th getter

        :param i: integer, i-th robot in simulation
        :return: Mx1 vector, m maximum-reach LiDAR scanning ranges of i-th robot
        """
        return self.__all_range_data[i, :]

    def set_shared_obs(self, i, j, arr_obstacle):
        """
        __shared_detected_pos @ i,j-th cell setter

        :param i: integer, the original robot's index
        :param j: integer, the neighbor's index
        :param arr_obstacle: list of 2x1 vector, list of detected obstacle in shared region
        """
        self.__shared_detected_pos[f'{i}_{j}'] = arr_obstacle

    def get_shared_obs(self, i, j):
        """
        __shared_detected_pos @ j,i-th cell getter

        :param i: integer, the original robot's index
        :param j: integer, the neighbor's index
        :return: list of 2x1 vector, list of detected obstacle in shared region
        """
        key = f'{i}_{j}'
        if key not in self.__shared_detected_pos:
            self.__shared_detected_pos[key] = np.zeros((0, 3))
        return self.__shared_detected_pos[key]
