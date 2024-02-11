import numpy as np

USE_QPSOLVERS = True

if USE_QPSOLVERS:
    from qpsolvers import solve_qp
else:
    import cvxopt


class cbf_si():
    def __init__(self, P=None, q=None, scale=1, scale_constraint=False):
        """
        Initialize the class for methods calling
        """
        self.is_scale_constraint = scale_constraint
        self.s = 0
        self.scale = scale
        self.scale_num = 1
        self.reset_cbf()

    def reset_cbf(self):
        """
        Empty the G and h in Gu* ≤ h
        """
        self.constraint_G = None
        self.constraint_h = None
        self.cbf_values = None

    def __set_constraint(self, G_mat, h_mat):
        """
        Add/Append rows to G and h in G @ u ≤ h

        :param G_mat: Nx2 matrix
        :param h_mat: Nx1 matrix
        """
        if self.constraint_G is None:
            self.constraint_G = G_mat
            self.constraint_h = h_mat
        else:
            self.constraint_G = np.append(self.constraint_G, G_mat, axis=0)
            self.constraint_h = np.append(self.constraint_h, h_mat, axis=0)

    def compute_safe_controller(self, u_nom, s_nom, P=None, q=None, weight=1.):
        """
        Compute u_star

        :param weight: scalar value, weight of v
        :param s_nom: scalar value, nominal v
        :param u_nom: 3x1 vector, nominal u
        :param P: 4x4 matrix, in x'Px + q'x + c
        :param q: 4x1 matrix, in x'Px + q'x + c
        :return u_star, s_star: 4x1 vector, optimal u + v
        """

        if (P is None) and (q is None):
            s_nom = s_nom[s_nom != 0]
            P, q = 2 * np.eye(3 + self.scale_num), -2 * np.hstack((u_nom, weight * s_nom))
            for i in range(self.scale_num):
                P[3 + i, 3 + i] = 2 * weight

        if self.constraint_G is not None:
            if USE_QPSOLVERS:
                sol = solve_qp(P, q, self.constraint_G, self.constraint_h,
                               solver="daqp")
                #   solver="quadprog")
                #   solver="proxqp")

                u_star = np.array([sol[0], sol[1], sol[2]])
                s_star = np.array(sol[3:3 + self.scale_num])

            else:
                G_mat = self.constraint_G.copy()
                h_mat = self.constraint_h.copy()
                # IMPLEMENTATION OF Control Barrier Function
                if self.is_scale_constraint:
                    for i in range(len(h_mat)):
                        G_mat[i] = self.constraint_G[i] / self.constraint_h[i]
                        h_mat[i] = 1.

                # Minimization
                P_mat = cvxopt.matrix(P.astype(np.double), tc='d')
                q_mat = cvxopt.matrix(q.astype(np.double), tc='d')
                # Resize the G and H into appropriate matrix for optimization
                G_mat = cvxopt.matrix(self.constraint_G.astype(np.double), tc='d')
                h_mat = cvxopt.matrix(self.constraint_h.astype(np.double), tc='d')
                # Solving Optimization
                cvxopt.solvers.options['show_progress'] = False
                sol = cvxopt.solvers.qp(P_mat, q_mat, G_mat, h_mat, verbose=False)

                if sol['status'] == 'optimal':
                    # Get solution + converting from cvxopt base matrix to numpy array
                    u_star = np.array([sol['x'][0], sol['x'][1], sol['x'][2]])
                    s_star = np.array(sol['x'][3])
                    # print('OPTIMAL:', s_star)
                else:
                    print('WARNING QP SOLVER' + ' status: ' + sol['status'] + ' --> use nominal instead')
                    u_star = u_nom.copy()
                    s_star = s_nom.copy()


        else:  # No constraints imposed
            u_star = u_nom.copy()
            s_star = s_nom.copy()

        self.s = s_star
        return u_star, s_star

    # ADDITION OF CONSTRAINTS
    # -----------------------------------------------------------------------------------------------------------
    def add_avoid_static_circle(self, pos, obs, ds, gamma=10, power=3):
        """
        Add equation related to circular obstacle avoidance

        :param pos: 2x1 vector, robot position
        :param obs: 2x1 vector, neighbor
        :param ds: scalar value, desired distance
        :param gamma: scalar value, coefficient of gamma function
        :param power: scalar value, degree of gamma function
        :return h_func: scalar values, safety estimations
        """
        # h = norm2(pos - obs)² - norm2(ds)² > 0
        vect = pos - obs
        h_func = np.power(np.linalg.norm(vect), 2) - np.power(ds, 2)
        # -(dh/dpos)^T u < gamma(h)
        vect_extended = np.hstack((-2 * vect.reshape((1, 3)), np.array([[0.] * self.scale_num])))
        self.__set_constraint(vect_extended, gamma * np.power(h_func, power).reshape((1, 1)))

        return h_func

    @staticmethod
    def regulate(angle):
        """
        Return radian angular value between -π → π

        :param angle: scalar value, radian angular input
        :return: scalar value, radian angular value between -π → π
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def is_between(angle, min_val, max_val):
        """
        Check if the angle is in the radian region of (min_val, max_val)
        π → -π leap checking involved
        :param angle: scalar value, radian angular input
        :param min_val: scalar value, minimum radian angular value
        :param max_val: scalar value, maximum radian angular value
        :return: boolean, True if in between, False otherwise
        """
        return (angle > min_val) & (angle < max_val) if (max_val > min_val) \
            else (angle > min_val) | (angle < max_val)

    @staticmethod
    def is_between_equal(angle, min_val, max_val):
        """
        Check if the angle is in the radian region of [min_val, max_val]
        π → -π leap checking involved

        :param angle: scalar value, radian angular input
        :param min_val: scalar value, minimum radian angular value
        :param max_val: scalar value, maximum radian angular value
        :return: boolean, True if in between, False otherwise
        """
        return (angle >= min_val) & (angle <= max_val) if (max_val > min_val) \
            else (angle >= min_val) | (angle <= max_val)

    def add_avoid_lidar_detected_obs(self, obs_pos, pos_i, kappa, ds, gamma=10, power=3):

        """
        Process the obstacle sensing data from robots and add constraints

        :param obs_pos: Nx3 array, stacked obstacle positions detected by i-th robot
        :param pos_i: 1x3 array, i-th robot position
        :param kappa: scalar value, kappa
        :param ds: scalar value, minimum distance to obstacle
        :param gamma: scalar value, coefficient of gamma function
        :param power: scalar value, degree of gamma function
        :return h_func: scalar values, safety estimations
        """

        # Default value of return if obs_pos is empty
        min_h = np.nan

        # Process the obstacle detected points
        if obs_pos.shape[0] > 0:
            # Identify obstacles position (in polar coordinate of world frame)
            vec_iobs = obs_pos - pos_i
            obst_range = np.linalg.norm(vec_iobs, axis=1)
            obst_angle = np.arctan2(vec_iobs[:, 1], vec_iobs[:, 0])

            # TRUE when closest to pos_i --> within (phi_ia + pi/2, phi_ib - pi/2)
            is_closest_i = self.is_between_equal(obst_angle, -3 * np.pi, 3 * np.pi)

            # Process further only if the set is not empty
            if np.sum(is_closest_i) > 0:
                # Compute Distance to Agent
                # Filter the correct shortest distance
                BIG_VALUE = np.max(obst_range) * 100
                dist_to_form = BIG_VALUE * np.ones(obst_angle.shape)  # BIG_VALUE by default for unuse
                dist_to_form[is_closest_i] = obst_range[is_closest_i]
                # Hybrid CBF for a set of closest obstacle only
                # --------------------------------------------------
                h_obs = dist_to_form ** 2 - ds ** 2
                min_h = np.min(h_obs)
                # Determine which to compute
                is_computed = h_obs < min_h + kappa

                gamma_h = gamma * np.power(min_h, power).reshape((1, 1))

                for i in np.where(is_computed & is_closest_i)[0]:
                    vect_extended = np.hstack((2 * vec_iobs[i, :].reshape((1, 3)), np.array([[0.] * self.scale_num])))
                    self.__set_constraint(vect_extended, gamma_h)

        return min_h

    def add_maintain_distance_with_scale(self, pos, obs, ds, max_dist, min_dist, self_scale,
                                         neigh_scale, epsilon, gamma=10, power=3):

        vect = pos - obs
        combined_scale = (self_scale + neigh_scale) / 2
        idx = 0

        # Upper distance
        # h = norm2( combined_scale * ds + epsilon )^2 - norm2( pos - obs )^2 ≥ 0
        h_fmu = np.power((combined_scale * ds + epsilon), 2) - np.power(np.linalg.norm(vect), 2)
        vect_u = np.hstack((2 * vect.reshape((1, 3)), np.zeros((1, self.scale_num))))
        vect_u[0, idx + 3] = - ds * (combined_scale * ds + epsilon)
        # -(dh/dpos)^T u < gamma(h)
        self.__set_constraint(vect_u, gamma * np.power(h_fmu, power).reshape((1, 1)))

        # Lower distance
        # h = norm2( pos - obs )^2 - norm2( combined_scale * ds - epsilon )^2 ≥ 0
        h_fml = np.power(np.linalg.norm(vect), 2) - np.power((combined_scale * ds - epsilon), 2)
        vect_l = np.hstack((-2 * vect.reshape((1, 3)), np.zeros((1, self.scale_num))))
        vect_l[0, idx + 3] = ds * (combined_scale * ds - epsilon)
        # -(dh/dpos)^T u < gamma(h)
        self.__set_constraint(vect_l, gamma * np.power(h_fml, power).reshape((1, 1)))

        # Upper scale
        # h = max_dist - combined_scale * ds ≥ 0
        common_vect = np.zeros((1, self.scale_num + 3))
        common_vect[0, idx + 3] = ds / 2
        h_scale_ceil = max_dist - combined_scale * ds
        self.__set_constraint(common_vect, gamma * np.power(h_scale_ceil, power).reshape((1, 1)))

        # Lower scale
        # h = combined_scale * ds - min_dist ≥ 0
        h_scale_floor = combined_scale * ds - min_dist
        self.__set_constraint(-common_vect, gamma * np.power(h_scale_floor, power).reshape((1, 1)))

        return h_fml, h_fmu, h_scale_floor, h_scale_ceil


    def update_additional_state(self, Ts):
        """
        Update scale values according to saved v and input timestep
        :param Ts: scalar value, input timestep
        """
        self.scale += Ts * self.s

    def get_additional_state(self):
        """
        Return all scale values
        :return: one-dimension array, epsilon values
        """
        return self.scale

    def add_velocity_bound(self, vel_limit):
        """
        Add velocity threshold as constraints
        :param vel_limit: scalar value
        """
        G = np.vstack((np.eye(3), -np.eye(3)))
        h = np.ones([6, 1]) * vel_limit
        self.__set_constraint(G, h)

    # TODO: add area with boundary