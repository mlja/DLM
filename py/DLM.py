import numpy as np
from scipy.special import factorial # fix this at some point

#
class DLM_ParticleTool():
    """
    Various kinds of useful matrices and functionalities for DLM-based applications
    Particle motion (random excitation imposed e.g. on velocity or acceleration)
    The paper "E. V. Stansfield: Introduction To Kalman Filters" is recommendable
    """
    
    # EMPTY CONSTRUCTOR
    def __init__(self):
        pass
    
    # Particle-type transition matrix (evolution matrix)
    @staticmethod
    def particle_transition(tau, dimension=3):
        d = dimension # rename
        phi = np.zeros([d,d])
        first_row = float(tau)**np.arange(d) # powers of tau
        first_row /= factorial( np.arange(d) ) # divide by factorials
        phi[0,:] = first_row
        for i in range(d-1):
            phi[i+1,:] = np.r_[0,phi[i,0:-1]] # push away last entry
        return phi # (d x d)
    
    # Particle-type covariance matrix
    @staticmethod
    def particle_covar(tau, sigma, dimension=3):
        d = dimension # rename
        powers = np.asfarray(np.arange(d)[::-1]) # [d-1,d-2,...,1,0]
        W = np.outer(float(tau)**powers,float(tau)**powers)
        W /= np.outer( factorial(powers), factorial(powers) )
        sum_term = np.ones([d,d])
        sum_term += np.reshape(np.kron(powers,np.ones(d)),[d,d])
        sum_term += np.reshape(np.kron(np.ones(d),powers),[d,d])
        W /= sum_term
        return float(tau*sigma**2)*W # (d x d)
    
    # Diagonal scaling of square matrix (enforcement of stationarity)
    @staticmethod
    def scale_diagonal(a, dimension=3):
        d = dimension # rename
        S = np.ones([d,d])
        for i in range(d):
            S[i,i] *= a
        return S # (d x d)

#
class DLM_Class():
    """
    West & Harrison, Bayesian Forecasting and Dynamic Models, 2nd Edition, Springer, 1999
    The general theory, definitions, updating schemes, etc., are located in Chapter 4
    """
    
    # allocate/declare matrices and vectors
    def __init__(self, n, r, D=np.float64): # n = state dim, r = observation dim
        self.m = np.zeros(n, dtype=D) # state (n x 1)
        self.a = np.zeros(n, dtype=D) # prediction (n x 1) of the next state
        self.f = np.zeros(r, dtype=D) # forecast (r x 1)
        self.C = np.eye(n, dtype=D) # posterior covariance
        self.R = np.zeros([n, n], dtype=D) # prediction covariance
        self.Q = np.zeros([r, r], dtype=D) # forecast covariance
        self.A = np.zeros([n, r], dtype=D) # adaption matrix (Kalman gain)
        self.F = np.ones([n, r], dtype=D) # design matrix
        self.G = np.eye(n, dtype=D) # evolution matrix
        self.V = np.eye(r, dtype=D) # observation covariance
        self.W = np.eye(n, dtype=D) # evolution covariance
        self._n = n
        self._r = r
    
    # use when both models are linear (evolution and observation)
    def iterate_DLM(self, Yt):
        self.a[:] = np.dot( self.G, self.m )
        self.R[:,:] = np.dot(self.G, np.dot(self.C, self.G.T)) + self.W
        self.f[:] = np.dot( self.F.T, self.a )
        self.Q[:,:] = np.dot(self.F.T, np.dot(self.R, self.F)) + self.V
        self.A[:,:] = np.array( np.dot(self.R, np.dot(self.F, np.matrix(self.Q).I)) )
        self.m[:] = self.a + np.dot( self.A, Yt-self.f )
        self.C[:,:] = np.dot( self.R, np.eye(self._n) - np.dot(self.F, self.A.T) )
        return
    
    # use when observation model is non-linear (but evolution is linear)
    def iterate_DnLM(self, Yt, evaluateF, jacobiF):
        self.a[:] = np.dot( self.G, self.m )
        self.R[:,:] = np.dot(self.G, np.dot(self.C, self.G.T)) + self.W
        self.f[:] = evaluateF(self.a) # nonlinear mapping in replace of F.T (from n to r variables)
        J = jacobiF(self.a) # (r x n), J plays the role of F.T (not F)
        self.Q[:,:] = np.dot(J, np.dot(self.R, J.T)) + self.V
        self.A[:,:] = np.array( np.dot(self.R, np.dot(J.T, np.matrix(self.Q).I)) )
        self.m[:] = self.a + np.dot( self.A, Yt-self.f )
        self.C[:,:] = np.dot( self.R, np.eye(self._n) - np.dot(J.T, self.A.T) )
        return

#