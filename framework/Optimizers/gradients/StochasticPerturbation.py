import abc
import copy
import pandas as pd
import numpy as np
import os
import sys
from .GradientApproximater import GradientApproximater
CWD = (os.getcwd())
CWD = CWD +'/../../'
sys.path.append(CWD)
print("This is current",os.getcwd())
from utils import InputData, InputTypes, randomUtils, mathUtils

from utils import InputData, InputTypes, randomUtils, mathUtils



"Author:--"

class StochasticPerturbation(GradientApproximater):
  
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
      new2 = copy.deepcopy(opt)
      delta = dh 

      new[optVar] = optValue + delta
      new2[optVar] = optValue - delta

      evalPoints.append(new)
      evalPoints.append(new2)
      evalInfo.append({'type': 'grad',
                       'optVar': optVar,
                       'delta': delta})

      evalInfo.append({'type': 'grad',
                       'optVar': optVar,
                       'delta': delta})

    
      
    return evalPoints, evalInfo

  def evaluate(self, opt, grads, infos, objVar):
    """
      Approximates gradient based on evaluated points.
      @ In, opt, dict, current opt point (normalized)
      @ In, grads, list(dict), evaluated neighbor points
      @ In, infos, list(dict), info about evaluated neighbor points
      @ In, objVar, string, objective variable
      @ Out, magnitude, float, magnitude of gradient
      @ Out, direction, dict, versor (unit vector) for gradient direction
    """

    gradient = {}

    objective = np.zeros((len(grads)))
    delta={}
    for g, pt in enumerate(grads):
      info = infos[g] 
      activeVar = info['optVar']
      delta[activeVar] = info['delta']


    
    
    for i,var in enumerate(grads[0].keys()):

      if var != objVar:

        gradient[var] = (-3*grads[2*i+1][objVar]+4*opt[objVar]-grads[2*i][objVar])/(2*delta[var])
    



    
    magnitude, direction, foundInf = mathUtils.calculateMagnitudeAndVersor(list(gradient.values()))
    direction = dict((var, float(direction[v])) for v, var in enumerate(gradient.keys()))
    return magnitude, direction, foundInf

  def numGradPoints(self):
    """
      Returns the number of grad points required for the method
    """

    return self.N*2
  

class Test():

    def Peturb(self, N):
        A=randomUtils.randPointsOnHypersphere(N)

        print(A)

        return(A)


B=Test()

print(B.Peturb)





    
  
  



