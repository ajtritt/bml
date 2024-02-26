import numpy as np

from abc import ABCMeta, abstractmethod

class Objective(metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, V_app, Q, V_fe_avg, Phi_S):
        pass

class DPhiSOverDVapp(Objective):

    _reductions = {
            'min': np.min,
            'max': np.max,
            'mean': np.mean,
            'median': np.median
    }

    def __init__(self, mode='mean'):
        if mode not in self._reductions:
            raise ValueError('Valid modes are min, max, mean, median')
        self.reduction = self._reductions[mode]

    def __call__(self, V_app, Q, V_fe_avg, Phi_S):
        Phi_S = [self.reduction(_) for _ in Phi_S]
        dPhiS_dVapp = (np.diff(Phi_S) / np.diff(V_app))
        response = np.max(dPhiS_dVapp)
        return response

class PhiS(Objective):

    _reductions = {
            'min': np.min,
            'max': np.max,
            'mean': np.mean,
            'median': np.median
    }

    def __init__(self, mode='mean'):
        if mode not in self._reductions:
            raise ValueError('Valid modes are min, max, mean, median')
        self.reduction = self._reductions[mode]

    def __call__(self, V_app, Q, V_fe_avg, Phi_S):
        Phi_S = [self.reduction(_) for _ in Phi_S]
        response = np.max(Phi_S)
        return response
