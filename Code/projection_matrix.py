import time
from typing import Tuple

import numpy as np
from scipy.linalg import rq
from scipy.optimize import least_squares


def objective_func(x: np.ndarray, **kwargs):
    """
    Calculates the difference in image (pixel coordinates) and returns 
    it as a 2*n_points vector

    Args: 
    -        x: numpy array of 11 parameters of P in vector form 
                (remember you will have to fix P_34=1) to estimate the reprojection error
    - **kwargs: dictionary that contains the 2D and the 3D points. You will have to
                retrieve these 2D and 3D points and then use them to compute 
                the reprojection error. 
                To get the 2D points, use kwargs['pts2d']
                To get the 3D points, use kwargs['pts3d']
    Returns:
    -     diff: A 2*N_points-d vector (1-D numpy array) of differences betwen 
                projected and actual 2D points

    """

    diff = None

    points_2d = kwargs['pts2d']
    points_3d = kwargs['pts3d']

    ##############################
    # TODO: Student code goes here
    z = x
    z = np.append(z,1)
    z = np.reshape(z,(3,4))
    projection_hat = projection(z, points_3d)
    diff = projection_hat - points_2d
    ##############################

    return diff.flatten()


def projection(P: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    """
        Computes projection from [X,Y,Z,1] in non-homogenous coordinates to
        (x,y) in non-homogenous image coordinates.

        Args:
        -  P: 3x4 projection matrix
        -  points_3d : n x 3 array of points [X_i,Y_i,Z_i]

        Returns:
        - projected_points_2d : n x 2 array of points in non-homogenous image coordinates
    """
    n,_ = np.shape(points_3d)
    projected_points_2d = np.zeros((n,2))

    assert points_3d.shape[1]==3

    ##############################
    # TODO: Student code goes here
    for i in range(n):
        denominator = (P[2][0]*points_3d[i][0] + P[2][1]*points_3d[i][1] + P[2][2]*points_3d[i][2] + P[2][3])
        projected_points_2d[i][0] = (P[0][0]*points_3d[i][0] + P[0][1]*points_3d[i][1] + P[0][2]*points_3d[i][2] + P[0][3])/ denominator
        projected_points_2d[i][1] = (P[1][0]*points_3d[i][0] + P[1][1]*points_3d[i][1] + P[1][2]*points_3d[i][2] + P[1][3])/ denominator

    ##############################

    return projected_points_2d

################# UPDATE function name to estimate_projection_matrix here or update in notebook #########################
def estimate_camera_matrix(pts2d: np.ndarray,
                           pts3d: np.ndarray,
                           initial_guess: np.ndarray) -> np.ndarray:
    '''
        Calls least_squares form scipy.least_squares.optimize and
        returns an estimate for the camera projection matrix

        Args:
        - pts2d: n x 2 array of known points (x_i, y_i) in image coordinates 
        - pts3d: n x 3 array of known points in 3D, (X_i, Y_i, Z_i, 1) 
        - initial_guess: 3x4 projection matrix initial guess

        Returns:
        - P: 3x4 estimated projection matrix 

        Note: Because of the requirements of scipy.optimize.least_squares
              you will have to pass the projection matrix P as a vector.
              Since we will fix P_34 to 1 you will not need to pass all 12
              matrix parameters. 

              You will also have to put pts2d and pts3d into a kwargs dictionary
              that you will add as an argument to least squares.

              We recommend that in your call to least_squares you use
              - method='lm' for Levenberg-Marquardt
              - verbose=2 (to show optimization output from 'lm')
              - max_nfev=50000 maximum number of function evaluations
              - ftol \
              - gtol  --> convergence criteria
              - xtol /
              - kwargs -- dictionary with additional variables 
                          for the objective function
    '''

    P = None

    start_time = time.time()

    kwargs = {'pts2d': pts2d,
              'pts3d': pts3d}


    ##############################
    # TODO: Student code goes here
    vectorizedP = initial_guess.flatten()
    vectorizedP = np.delete(vectorizedP,-1)
    OptimizedResult = least_squares(objective_func,vectorizedP, method='lm',verbose=2, kwargs= kwargs)
    P = OptimizedResult.x
    P = np.append(P,1)
    P = np.reshape(P, (3,4))
    ##############################

    print("Time since optimization start", time.time() - start_time)

    return P


def decompose_camera_matrix(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
        Decomposes the camera matrix into the K intrinsic and R rotation matrix

        Args:
        -  P: 3x4 numpy array projection matrix

        Returns:

        - K: 3x3 intrinsic matrix (numpy array)
        - R: 3x3 orthonormal rotation matrix (numpy array)

        hint: use scipy.linalg.rq()
    '''
    K = None
    R = None

    ##############################
    # TODO: Student code goes here
    M = P[:,0:3]
    K,R = rq(M)
    ##############################

    return K, R


def calculate_camera_center(P: np.ndarray,
                            K: np.ndarray,
                            R_T: np.ndarray) -> np.ndarray:
    """
    Returns the camera center matrix for a given projection matrix.

    Args:
    -   P: A numpy array of shape (3, 4) representing the projection matrix

    Returns:
    -   cc: A numpy array of shape (1, 3) representing the camera center
            location in world coordinates
    """

    cc = None

    ##############################
    # TODO: Student code goes here

    I_Xc = (np.linalg.inv(R_T)@np.linalg.inv(K))@P
    cc = -I_Xc[:,3]
    ##############################

    return cc
