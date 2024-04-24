import numpy as np

USE_QPSOLVERS = True

if USE_QPSOLVERS:
    from qpsolvers import solve_qp, Problem, solve_problem
else:
    import cvxopt


class cbf_si():
    def __init__(self, id, P=None, q=None, own_eps_ij=np.array([]), scale_constraint=False):
        """
        Initialize the class for methods calling
        """
        self.is_scale_constraint = scale_constraint

        self._id = id
        self._own_eps_ij = own_eps_ij # initialize own epsilon
        # Identify number of neighbours and the size of additional variables
        self.id_neighbours = np.where(own_eps_ij > 0)[0]
        self.eps_num = len(self.id_neighbours)
        self.v = np.zeros(self.eps_num)
        
        # Total number of decision variables
        self._var_num = 3 + self.eps_num

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

    def compute_safe_controller(self, u_nom, v_nom=np.array([]), P=None, q=None, weight=1., speed_limit = None):
        """
        Compute u_star

        :param weight: scalar value, weight of v
        :param v_nom: scalar value, nominal v
        :param u_nom: 3x1 vector, nominal u
        :param P: 4x4 matrix, in x'Px + q'x + c
        :param q: 4x1 matrix, in x'Px + q'x + c
        :return u_star, v_star: 4x1 vector, optimal u + v
        """

        if (P is None) and (q is None):
            P, q = 2 * np.eye(self._var_num), np.zeros(self._var_num)
            q[:3] = -2*u_nom

            if self.eps_num > 0:
                for i in range(self.eps_num):
                    P[3 + i, 3 + i] = 2 * weight
                    q[3+i] = -2*weight*v_nom[self.id_neighbours[i]]



        if self.constraint_G is not None:
            if USE_QPSOLVERS:
                def_ublb = np.inf
                lb = np.ones(self._var_num)*(-def_ublb)
                ub = np.ones(self._var_num)*(def_ublb)

                if speed_limit is not None:
                    array_limit = np.ones(3)* speed_limit
                    lb[:3], ub[:3] = -array_limit, array_limit

                # SCALING PROCESS
                G_mat = self.constraint_G.copy()
                h_mat = self.constraint_h.copy()
                # IMPLEMENTATION OF Control Barrier Function
                if self.is_scale_constraint: # WARNING: scaling does not work
                    for i in range(len(h_mat)):
                        # print(self.constraint_G[i], self.constraint_h[i,0])
                        G_mat[i] = self.constraint_G[i] / self.constraint_h[i,0]
                        h_mat[i] = self.constraint_h[i] / self.constraint_h[i,0]

                opt_tolerance = 1e-8

                qp_problem = Problem(P, q, G_mat, h_mat, lb = lb, ub = ub,)
                qp_problem.check_constraints()
                # qp_problem.cond()
                # solution = solve_problem(qp_problem, solver="daqp")
                solution = solve_problem(qp_problem, solver="daqp", dual_tol=opt_tolerance, primal_tol=opt_tolerance)
                # solution = solve_problem(qp_problem, solver="clarabel")
                # solution = solve_problem(qp_problem, solver="clarabel", 
                #          tol_feas=1e-9, tol_gap_abs=1e-9, tol_gap_rel=0)
                # solution = solve_problem(qp_problem, solver="quadprog")
                sol = solution.x

                # sol = solve_qp(P, q, self.constraint_G, self.constraint_h,
                #             lb = lb, ub = ub,
                #             solver="daqp")
                #   solver="quadprog")
                #   solver="proxqp")


                if sol is None:
                    print(self._id, 'WARNING QP SOLVER [no solution] stopping instead')
                    u_star = np.array([0., 0., 0.])
                    v_star = 0*v_nom.copy()

                    # print('constraints:', h_mat)
                    # print('G:', G_mat)

                    # print('P:', P, 'q:', q)
                    # print('G:', G_mat)
                    # print('h:', h_mat)
                    # qp_problem.save('no_solution')
                    # exit()


                else:
                    if not solution.is_optimal(opt_tolerance):
                        print(self._id, 'WARNING QP SOLVER [not optimal] stopping instead')
                        u_star = np.array([0., 0., 0.])
                        v_star = 0*v_nom.copy()
                            
                        # check_val = G_mat@sol
                        # print('constraints:', check_val, h_mat[:, 0])
                        
                        # print('P:', P, 'q:', q)
                        # print('G:', G_mat)
                        # print('h:', h_mat)
                        # qp_problem.save('not_optimal')
                        # exit()


                    else:
                        u_star = np.array([sol[0], sol[1], sol[2]])
                        v_star = np.array(sol[3:])

                        # check_val = G_mat@sol
                        # print('constraints:', check_val, h_mat[:, 0])
                        # print('G:', G_mat)



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
                # Resize the G and H into appropriate matrix  -5.2076950637256205e-05 2.5293629629119387e-08 0.00072for optimization
                G_mat = cvxopt.matrix(self.constraint_G.astype(np.double), tc='d')
                h_mat = cvxopt.matrix(self.constraint_h.astype(np.double), tc='d')
                # Solving Optimization
                cvxopt.solvers.options['show_progress'] = False
                sol = cvxopt.solvers.qp(P_mat, q_mat, G_mat, h_mat, verbose=False)

                if sol['status'] == 'optimal':
                    # Get solution + converting from cvxopt base matrix to numpy array
                    u_star = np.array([sol['x'][0], sol['x'][1], sol['x'][2]])
                    v_star = np.array(sol['x'][3:])
                    # print('OPTIMAL:', v_star)
                else:
                    print('WARNING QP SOLVER' + ' status: ' + sol['status'] + ' --> use nominal instead')
                    u_star = u_nom.copy()
                    v_star = v_nom.copy()


        else:  # No constraints imposed
            u_star = u_nom.copy()
            v_star = v_nom.copy()

        self.v = v_star.flatten()
        return u_star, v_star


    # ADDITION OF CONSTRAINTS
    # -----------------------------------------------------------------------------------------------------------
    def add_avoid_static_circle(self, pos, obs, ds, gamma=10, power=3):
        """idx
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
        vect_extended = np.zeros( (1, self._var_num) )
        vect_extended[0, :3] = -2 * vect
        # nothing is assigned for the eps variable

        self.__set_constraint(vect_extended, gamma * np.power(h_func, power).reshape((1, 1)))

        return h_func
    

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
        data_num = obs_pos.shape[0]
        if data_num > 0:
            # Identify obstacles position (in polar coordinate of world frame)
            vec_iobs = obs_pos - pos_i
            obst_range = np.linalg.norm(vec_iobs, axis=1)

            # Hybrid CBF for a set of closest obstacle only
            # --------------------------------------------------
            h_obs = obst_range ** 2 - ds ** 2
            min_h = np.min(h_obs)
            # Determine which to computeidx
            is_computed = h_obs < min_h + kappa

            gamma_h = gamma * np.power(min_h, power).reshape((1, 1))

            for i in range(data_num):
                if is_computed[i]:
                    vect_extended = np.zeros( (1, self._var_num) )
                    vect_extended[0, :3] = 2 * vec_iobs[i, :]
                    # nothing is assigned for the eps variable

                    self.__set_constraint(vect_extended, gamma_h)

        return min_h


    def add_maintain_distance_with_distinct_epsilon(self, pos, obs, ds, max_epsilon, self_eps,
                                                    neigh_eps, neigh_id, 
                                                    gamma_fm=10, power_fm=3, 
                                                    gamma_eps=10, power_eps=3 ):
        """
        Add equations related to connectivity in formation

        :param pos: 2x1 vector, robot position
        :param obs: 2x1 vector, neighbor
        :param ds: scalar value, desired distance
        :param neigh_eps: scalar value, neighbor epsilon
        :param max_epsilon: scalar value, tolerance
        :param self_eps: scalar value, self epsilon
        :param neigh_id:
        :param gamma: scalar value, coefficient of gamma function
        :param power: scalar value, degree of gamma function
        :return h_func_l, h_func_u: scalar values, safety estimations
        """
        vect = pos - obs
        combined_eps = self_eps + neigh_eps
        idx = np.where(self.id_neighbours == neigh_id)[0]

        # Upper distance
        # h = norm2( ds + max_epsilon )^2 - norm2( pos - obs )^2 ≥ 0
        h_fmu = np.power((ds + combined_eps), 2) - np.power(np.linalg.norm(vect), 2)

        vect_u = np.zeros( (1, self._var_num) )
        vect_u[0, :3] = 2 * vect
        vect_u[0, 3+idx] = -2 * (combined_eps + ds)
        # -(dh/dpos)^T u < gamma(h)
        self.__set_constraint(vect_u, gamma_fm * np.power(h_fmu, power_fm).reshape((1, 1)))

        # Lower distance
        # h = norm2( pos - obs )^2 - norm2( min(ds - max_epsilon, min_dist) )^2 ≥ 0
        h_fml = np.power(np.linalg.norm(vect), 2) - np.power((ds - combined_eps), 2)

        vect_l = np.zeros( (1, self._var_num) )
        vect_l[0, :3] = -2 * vect
        vect_l[0, 3+idx] = -2 * (ds - combined_eps)
        # -(dh/dpos)^T u < gamma(h)
        self.__set_constraint(vect_l, gamma_fm * np.power(h_fml, power_fm).reshape((1, 1)))

        # Upper epsilon
        # h = ε_bar - ε_i - ε_j ≥ 0
        h_eps_ceil = max_epsilon - self_eps - neigh_eps

        common_vect = np.zeros( (1, self._var_num) )
        common_vect[0, 3+idx] = 1
        # -(dh/dpos)^T u < gamma(h)
        self.__set_constraint(common_vect, gamma_eps * np.power(h_eps_ceil, power_eps).reshape((1, 1)))

        # Lower epsilon
        # h = ε_i ≥ 0
        h_eps_floor = self_eps
        self.__set_constraint(-common_vect, gamma_eps * np.power(h_eps_floor, power_eps).reshape((1, 1)))

        return h_fml, h_fmu, h_eps_floor, h_eps_ceil


    def add_maintain_flexible_distance(self, pos, obs, ds, max_epsilon, self_eps,
                                        neigh_eps, neigh_id, gamma=10, power=3):
        """
        Add equations related to connectivity in formation

        :param pos: 2x1 vector, robot position
        :param obs: 2x1 vector, neighbor
        :param ds: scalar value, desired distance
        :param neigh_eps: scalar value, neighbor epsilon
        :param max_epsilon: scalar value, tolerance
        :param self_eps: scalar value, self epsilon
        :param neigh_id:
        :param gamma: scalar value, coefficient of gamma function
        :param power: scalar value, degree of gamma function
        :return h_func_l, h_func_u: scalar values, safety estimations
        """
        vect = pos - obs
        combined_eps = self_eps + neigh_eps
        idx = np.where(self.id_neighbours == neigh_id)[0]
        
        # Main parts of computation
        dist_ij = np.linalg.norm(vect)
        dist_ij_ds = dist_ij - ds
        fm = (combined_eps**2) - (dist_ij_ds**2) # Stay within ds +- eps
        em = (max_epsilon - combined_eps)*self_eps*neigh_eps # bound eps

        # # COMBINED VERSION
        # h_fm = fm*em
        # # -(dh/dpos)^T u < gamma(h)
        # vect_u = np.zeros((1, 3 + self.eps_num))
        # vect_u[0, :3] = -2*(dist_ij_ds*em/dist_ij)*vect # dh / dpi
        # vect_u[0, 3+idx] = (2*combined_eps*em) + fm*neigh_eps*(max_epsilon - (2*self_eps) - neigh_eps) # dh / deps_ij

        # gh = gamma * np.power(h_fm, power).reshape((1, 1))
        # self.__set_constraint(-vect_u, gh)

        # return h_fm, fm, em, -vect_u, gh

        # SPLIT VERSION
        # formation maintenance
        vect_u_fm = np.zeros((1, 3 + self.eps_num))
        vect_u_fm[0, :3] = -2*(dist_ij_ds/dist_ij)*vect # dfm / dpi
        vect_u_fm[0, 3+idx] = (2*combined_eps) # dfm / deps_ij
        gh_fm = gamma * np.power(fm, power).reshape((1, 1))
        self.__set_constraint(-vect_u_fm, gh_fm)

        # epsilon maintenance
        vect_u_em = np.zeros((1, 3 + self.eps_num))
        vect_u_em[0, 3+idx] = neigh_eps*(max_epsilon - (2*self_eps) - neigh_eps) # dem / deps_ij
        gh_em = gamma * np.power(em, power).reshape((1, 1))
        self.__set_constraint(-vect_u_em, gh_em)

        return fm, em


    def update_additional_state(self, Ts):
        """
        Update epsilon values according to saved v and input timestep
        :param Ts: scalar value, input timestep
        """
        # for idx, v in enumerate(self.v.tolist()):
        #     self._own_eps_ij[self.id_neighbours[idx]] += Ts * v
        for i in range(self.eps_num):
            self._own_eps_ij[self.id_neighbours[i]] += Ts * self.v[i]

    def get_additional_state(self):
        """
        Return all epsilon values
        :return: one-dimension array, epsilon values
        """
        return self._own_eps_ij[:]

    def add_velocity_bound(self, vel_limit):
        """
        Add velocity threshold as constraints
        :param vel_limit: scalar value
        """
        G = np.vstack((np.eye(3), -np.eye(3)))
        h = np.ones([6, 1]) * vel_limit
        self.__set_constraint(G, h)

    # TODO: add area with boundary