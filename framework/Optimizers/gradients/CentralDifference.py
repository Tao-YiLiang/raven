import abc
import copy
import pandas as pd
from utils import InputData, InputTypes, randomUtils, mathUtils

from utils import InputData, InputTypes, randomUtils, mathUtils

from .GradientApproximater import GradientApproximater

from .FiniteDifference import FiniteDifference

class CentralDifference(FiniteDifference):

    i = 0

    def derivative(self,x,y,delta, grads,objVar,infos):
      delta0 = infos[0]['delta']
      delta1 = infos[1]['delta']
      if self.i == 10:
        print("This is infos",infos[0], infos[1],grads[0][objVar],grads[1][objVar])
        #aaa
      
      if delta0 <0:
        delta0 =-1*delta0
      if delta1<0:
        delta1 =-1*delta1

      central = (-3*grads[0][objVar] + 4*opt[objVar] - grads[1][objVar])/(delta0+delta1)

      central = (mathUtils.diffWithInfinites(grads[0][objVar],grads[1][objVar]))/(delta0+delta1)
      self.i+=1
      return central
        


