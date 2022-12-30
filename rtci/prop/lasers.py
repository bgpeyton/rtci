import numpy as np

class delta: # delta pulse
    def __init__(self, F_str, center=0.0, tol=1e-7):
        self.F_str = F_str
        self.center = center
        self.tol = tol
    def __call__(self, t):
        if abs(t - self.center) <= self.tol:
            pulse = self.F_str * 1.0
        else:
            pulse = 0
        return pulse

class cos2: #cosine-squared 
    def __init__(self, F_str, sigma=0.0, center=0.0, omega=0.0):
        self.F_str = F_str
        self.sigma = sigma
        self.center = center
        self.omega = omega
    def _envelope(self,t):
        return np.cos(np.pi/2/self.sigma * (t - self.center))**2
    def __call__(self,t):
        if (t<(self.center-self.sigma) or (t>(self.center+self.sigma))):
            return 0
        else:
            p = np.cos(self.omega*(t-self.center))
            return self.F_str*self._envelope(t)*p
