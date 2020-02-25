import abc
import copy
import pandas as pd
from utils import InputData, InputTypes, randomUtils, mathUtils

from utils import InputData, InputTypes, randomUtils, mathUtils

from .GradientApproximater import GradientApproximater

from .FiniteDifference import FiniteDifference

class Stochastic(FiniteDifference):


  def derivative(self, x,y, delta,grads,objVar,infos):
    
    lossDiff = np.atleast_1d(mathUtils.diffWithInfinites(x,y))

    return lossDiff/delta
