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
