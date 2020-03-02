import abc
import copy
import pandas as pd
import numpy as np
import os
import sys
raven_path= '/Users/gaira/Optimizers/raven/framework/utils/'
sys.path.append(raven_path)
from utils import InputData, InputTypes, randomUtils, mathUtils

from utils import InputData, InputTypes, randomUtils, mathUtils

from .GradientApproximater import GradientApproximater

"Author:--"

class CentralDifference(GradientApproximater):
  """
    Enables gradient estimation via central differencing
  """
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
      neg = copy.deepcopy(opt)
      pos = copy.deepcopy(opt)
      delta = dh
      neg[optVar] = optValue - delta
      pos[optVar] = optValue + delta

      evalPoints.append(neg)
      evalInfo.append({'type': 'grad',
                      'optVar': optVar,
                      'delta': delta,
                      'side': 'negative'})

      evalPoints.append(pos)
      evalInfo.append({'type': 'grad',
                       'optVar': optVar,
                       'delta': delta})
    return evalPoints, evalInfo

  def evaluate(self, opt, grads, infos, objVar):
    """
      Approximates gradient based on 2-point stencil central difference of evaluated points.
      @ In, opt, dict, current opt point (normalized)
      @ In, grads, list(dict), evaluated neighbor points
      @ In, infos, list(dict), info about evaluated neighbor points
      @ In, objVar, string, objective variable
      @ Out, magnitude, float, magnitude of gradient
      @ Out, direction, dict, versor (unit vector) for gradient direction
    """
    gradient = {}

    for i in range(len(grads)):
      for var in infos[i]['optVar']:
        for j in range(i+1,len(grads)):
          if infos[j]['optVar']==var:
            pair = [grads[i][var],grads[j][var]]
            ind = sorted(range(len(pair)), key=lambda k: pair[k])
            if ind == sorted(ind):
              backward,forward = i,j
            else:
              backward,forward = j,i
            gradient[var] = (-3*grads[backward][objVar]+4*opt[objVar]-grads[forward][objVar])/(2*infos[i]['delta'])

          else:
             continue
        
    magnitude, direction, foundInf = mathUtils.calculateMagnitudeAndVersor(list(gradient.values()))
    direction = dict((var, float(direction[v])) for v, var in enumerate(gradient.keys()))
    return magnitude, direction, foundInf

  def numGradPoints(self):
    """
      Returns the number of grad points required for the method
    """
    return self.N*2





