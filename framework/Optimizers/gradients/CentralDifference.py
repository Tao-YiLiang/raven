import abc
import copy
import pandas as pd
import numpy as np
from utils import InputData, InputTypes, randomUtils, mathUtils

from utils import InputData, InputTypes, randomUtils, mathUtils

from .GradientApproximater import GradientApproximater

from .FiniteDifference import FiniteDifference

class CentralDifference(FiniteDifference):
  
  #i=0
  def chooseEvaluationPoints(self, opt, stepSize):
    """
      Determines new point(s) needed to evaluate gradient
      @ In, opt, dict, current opt point (normalized)
      @ In, stepSize, float, distance from opt point to sample neighbors
      @ Out, evalPoints, list(dict), list of points that need sampling
      @ Out, evalInfo, list(dict), identifying information about points
    """
    dh = self._proximity * stepSize
    evalPoints = []
    evalInfo = []


    for o, optVar in enumerate(self._optVars):
 
      optValue = opt[optVar]
      new = copy.deepcopy(opt)
      delta = dh #* directions[o]

      new[optVar] = optValue + delta
      evalPoints.append(new)
      evalInfo.append({'type': 'grad',
                       'optVar': optVar,
                       'delta': delta})
      print("This is evalpoints cdf",evalPoints)
    return evalPoints, evalInfo

  def derivative(self, x,y, delta,grads,objVar,infos):
    #print("This is self.N",self.N)
  
    delta0 = infos[0]['delta']
    delta1 = infos[1]['delta']
        
    #lossDiff = np.atleast_1d(mathUtils.diffWithInfinites(x,y))
    central = (-3*grads[0][objVar] + 4*y - grads[1][objVar])/(delta0+delta1)
    #central = (mathUtils.diffWithInfinites(grads[0][objVar],grads[1][objVar]))/(delta0+delta1)

    return central

  def numGradPoints(self):
    """
      Returns the number of grad points required for the method
    """
    #print("This is self.N",self.N*2)
    
    #aaa
    #self.N+=self.N
    return self.N
  
  



