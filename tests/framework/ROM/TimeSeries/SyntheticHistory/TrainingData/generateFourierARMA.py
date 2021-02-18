# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
  Generates training data in the form of Fourier signals.
"""
import numpy as np
from generators import fourier, arma, toFile

plot = False

############
# "A" - simple superimposition
pivot = np.arange(1000)/10.

amps = [8, 10, 12]
periods = [2, 5, 10]
phases = [0, np.pi/4, np.pi]
intercept = 42
f = fourier(amps, periods, phases, pivot, mean=intercept)

slags = [0.4, 0.2]
nlags = [0.3, 0.2, 0.1]
a0, _ = arma(slags, nlags, pivot, plot=False)
slags = [0.5, 0.3]
nlags = [0.1, 0.05, 0.01]
a1, _ = arma(slags, nlags, pivot, plot=False)

s0 = f + a0
s1 = f + a1

if plot:
  import matplotlib.pyplot as plt
  fig, ax = plt.subplots()
  ax.plot(pivot, s0, '.-', label='0')
  ax.plot(pivot, s1, '.-', label='1')
  ax.legend()
  plt.show()

out = np.zeros((len(pivot), 3))
out[:, 0] = pivot
out[:, 1] = f + a0
out[:, 2] = f + a1
toFile(out, 'FourierARMA_A')

############
# "cluster" - distinct patterns
plot = True

pivot = np.arange(1000)/10.
signal = np.zeros(len(pivot))
subpivot = np.arange(100) / 10.

#     starts       0 1 2 3 4  5 6 7 8 9
# cluster pattern: A A B B C, C B C B A
Aspots = (slice(000, 100, None), slice(100, 200, None), slice(900, 1000, None))
Bspots = (slice(200, 300, None), slice(300, 400, None), slice(600, 700, None), slice(800, 900, None))
Cspots = (slice(400, 500, None), slice(500, 600, None), slice(700, 800, None))

np.random.seed(42)
def randomize(ary, pct=0):
  pctNoise = (2 * np.random.random(len(ary)) - 1) * pct
  return (1.0 + pctNoise) * ary

## pattern A
amps = np.asarray([2, 3, 4])
periods = [5, 10, 20]
phases = [np.pi/4]*3
intercept = 3.141592653589793238
slags = np.asarray([0.4])
nlags = np.asarray([0.1, 0.05])

for spot in Aspots:
  a = randomize(amps)
  s = randomize(slags)
  n = randomize(nlags)
  fA = fourier(a, periods, phases, subpivot, mean=intercept)
  aA, _ = arma(s, n, subpivot, plot=False)
  signal[spot] = fA + aA

## pattern B
amps = np.asarray([5, 10])
periods = [10, 20]
phases = [np.pi/4]*2
intercept = 42
slags = np.asarray([0.2])
nlags = np.asarray([0.2, 0.1])

for spot in Bspots:
  a = randomize(amps)
  s = randomize(slags)
  n = randomize(nlags)
  fB = fourier(a, periods, phases, subpivot, mean=intercept)
  aB, _ = arma(s, n, subpivot, plot=False)
  signal[spot] = fB + aB

## pattern C
amps = np.asarray([20, 30])
periods = [5, 20]
phases = [np.pi/4]*2
intercept = 100
slags = np.asarray([0.01])
nlags = np.asarray([0.05, 0.01])

for spot in Cspots:
  a = randomize(amps)
  s = randomize(slags)
  n = randomize(nlags)
  fC = fourier(a, periods, phases, subpivot, mean=intercept)
  aC, _ = arma(s, n, subpivot, plot=False)
  signal[spot] = fC + aC

if plot:
  import matplotlib.pyplot as plt
  fig, ax = plt.subplots()
  ax.plot(pivot, signal, '.-')
  plt.show()

out = np.zeros((len(pivot), 2))
out[:, 0] = pivot
out[:, 1] = signal
toFile(out, 'FourierARMA_cluster', pivotName='time')
