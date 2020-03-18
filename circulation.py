import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class Circulation:
    """
    Model of systemic circulation from Ferreira et al. (2005), A Nonlinear State-Space Model
    of a Combined Cardiovascular System and a Rotary Pump, IEEE Conference on Decision and Control.
    """

    def __init__(self, HR, Emax, Emin):
        self.set_heart_rate(HR)

        self.Emin = Emin
        self.Emax = Emax
        self.non_slack_blood_volume = 250 # ml

        self.R1 = 1.0 # between .5 and 2
        self.R2 = .005
        self.R3 = .001
        self.R4 = .0398

        self.C2 = 4.4
        self.C3 = 1.33

        self.L = .0005

    def set_heart_rate(self, HR):
        """
        Sets several related variables together to ensure that they are consistent.
        :param HR: heart rate (beats per minute)
        """
        self.HR = HR
        self.tc = 60/HR
        self.Tmax = .2+.15*self.tc # contraction time

    def get_derivative(self, t, x):
        """
        :param t: time
        :param x: state variables [ventricular pressure; atrial pressure; arterial pressure; aortic flow]
        :return: time derivatives of state variables
        """
        """
        WRITE CODE HERE
        Implement this by deciding whether the model is in a filling, ejecting, or isovolumic phase and using 
        the corresponding dynamic matrix. 
         
        As discussed in class, be careful about starting and ending the ejection phase. One approach is to check 
        whether the flow is >0, and another is to check whether x1>x3, but neither will work. The first won't start 
        propertly because flow isn't actually updated outside the ejection phase. The second won't end properly 
        because blood inertance will keep the blood moving briefly up the pressure gradient at the end of systole. 
        If the ejection phase ends in this time, the flow will remain non-zero until the next ejection phase. 
        """
        ventricular_pressure = x[0]
        atrial_pressure = x[1]
        arterial_pressure = x[2]
        aortic_flow = x[3]

        # filling phase occurs when x[1] > x[0]
        # ejection phase pccurs when x[3] > 0 or x[0] > x[2]
        # else isovolumic phase occurs
        if atrial_pressure > ventricular_pressure:
            A = self.filling_phase_dynamic_matrix(t)
        elif aortic_flow > 0 or ventricular_pressure > arterial_pressure:
            A = self.ejection_phase_dynamic_matrix(t)
        else:
            A = self.isovolumic_phase_dynamic_matrix(t)
        
        return np.matmul(A, x)

    def isovolumic_phase_dynamic_matrix(self, t):
        """
        :param t: time (s; needed because elastance is a function of time)
        :return: A matrix for isovolumic phase
        """
        el = self.elastance(t)
        del_dt = self.elastance_finite_difference(t)
        return [[del_dt/el, 0, 0, 0],
             [0, -1/(self.R1*self.C2), 1/(self.R1*self.C2), 0],
             [0, 1/(self.R1*self.C3), -1/(self.R1*self.C3), 0],
             [0, 0, 0, 0]]

    def ejection_phase_dynamic_matrix(self, t):
        """
        :param t: time (s)
        :return: A matrix for filling phase
        """
        el = self.elastance(t)
        del_dt = self.elastance_finite_difference(t)
        return [[del_dt/el, 0, 0, -el],
                [0, -1/(self.R1*self.C2), 1/(self.R1*self.C2), 0],
                [0, 1/(self.R1*self.C3), -1/(self.R1*self.C3), 1/self.C3],
                [1/self.L, 0, -1/self.L, -(self.R3+self.R4)/self.L]]

    def filling_phase_dynamic_matrix(self, t):
        """
        :param t: time (s)
        :return: A matrix for filling phase
        """
        el = self.elastance(t)
        del_dt = self.elastance_finite_difference(t)
        return [[del_dt/el-el/self.R2, el/self.R2, 0, 0],
                [1/(self.R2*self.C2), -(self.R1+self.R2)/(self.C2*self.R1*self.R2), 1/(self.C2*self.R1), 0],
                [0, 1/(self.R1*self.C3), -1/(self.R1*self.C3), 0],
                [0, 0, 0, 0]]

    def elastance(self, t):
        """
        :param t: time (needed because elastance is a function of time)
        :return: time-varying elastance
        """
        tn = self._get_normalized_time(t)
        En = 1.55 * np.power(tn/.7, 1.9) / (1 + np.power(tn/.7, 1.9)) / (1 + np.power(tn/1.17, 21.9))
        return (self.Emax-self.Emin)*En + self.Emin

    def elastance_finite_difference(self, t):
        """
        Calculates finite-difference approximation of elastance derivative. In class I showed another method
        that calculated the derivative analytically, but I've removed it to keep things simple.
        :param t: time (needed because elastance is a function of time)
        :return: finite-difference approximation of time derivative of time-varying elastance
        """
        dt = .0001
        forward_time = t + dt
        backward_time = max(0, t - dt) # small negative times are wrapped to end of cycle
        forward = self.elastance(forward_time)
        backward = self.elastance(backward_time)
        return (forward - backward) / (2*dt)

    def simulate(self, total_time):
        """
        :param total_time: seconds to simulate
        :return: time, state (times at which the state is estimated, state vector at each time)
        """
        # Start the simulation with all the blood in the atrium
        intial_cond = [0, self.non_slack_blood_volume/self.C2, 0, 0]
        time_dur = (0, total_time)

        def f(t, x):
            return self.get_derivative(t,x)

        sol = solve_ivp(f, time_dur, intial_cond, max_step=0.01)
        return sol.t, sol.y.T

    def _get_normalized_time(self, t):
        """
        :param t: time
        :return: time normalized to self.Tmax (duration of ventricular contraction)
        """
        return (t % self.tc) / self.Tmax

    # Question 3 Method Definition
    def left_ventricular_blood_vol(self, t, x):
        """
        :param t: time
        :param x: state variables [ventricular pressure; atrial pressure; arterial pressure; aortic flow]
        :return: left ventricular blood volume
        """
        # Assumption stated in assignment: Slack Volume is 20mL
        slack_vol = 20
        ventricular_pressure = x[:,0]
        blood_vol = ventricular_pressure / self.elastance(t) + slack_vol
        return blood_vol


def pressure_graphs(new_model, time, curr_state):
    aortic_press = curr_state[:, 2] + curr_state[:, 3] * new_model.R4

    plt.title('State of Circulation over Time')
    plt.plot(time, curr_state[:, 0], 'r', label='Ventricular Pressure')
    plt.plot(time, curr_state[:, 1], 'g', label='Atrial Pressure')
    plt.plot(time, curr_state[:, 2], 'b', label='Arterial Pressure')
    plt.plot(time, aortic_press, 'c', label='Aortic Pressure')
    plt.xlabel('Time (s)')
    plt.ylabel('State')
    plt.legend(loc='upper left')
    plt.show()

# Question 1
new_model = Circulation(75, 2.0, 0.06)

# Question 2    
t, x = new_model.simulate(5)
pressure_graphs(new_model, t, x)

# Question 3
print('Left Ventricular Blood Volumes: {}'.format(new_model.left_ventricular_blood_vol(t, x)))
