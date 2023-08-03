import numpy as np

from abc import ABCMeta, abstractmethod

class Objective(metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, V_app, Q, V_fe_avg, Phi_S):
        pass

class DPhiSOverDVapp(Objective):

    def __call__(self, V_app, Q, V_fe_avg, Phi_S):
        dPhiS_dVapp = (np.diff(Phi_S) / np.diff(V_app))
        response = np.max(dPhiS_dVapp)
        return response
