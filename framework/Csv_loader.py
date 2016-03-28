"""
Created on Feb 7, 2013
@author: alfoa
This python module performs the loading of
data from csv files
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
if not 'xrange' in dir(__builtins__): xrange = range
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import numpy as np
from scipy.interpolate import interp1d
import copy
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
import utils
import MessageHandler
#Internal Modules End--------------------------------------------------------------------------------

class CsvLoader(MessageHandler.MessageUser):
  """
    Class aimed to load the CSV files
  """
  def __init__(self,messageHandler):
    """
      Constructor
      @ In, messageHandler, MessageHandler, the message handler
      @ Out, None
    """
    self.allOutParam      = False # all output parameters?
    self.fieldNames        = []    #
    self.allFieldNames    = []
    self.type               = 'CsvLoader'
    self.printTag           = self.type
    self.messageHandler     = messageHandler

  def loadCsvFile(self,myFile):
    """
      Function to load a csv file into a numpy array (2D)
      It also retrieves the headers
      The format of the csv must be:
      STRING,STRING,STRING,STRING
      FLOAT ,FLOAT ,FLOAT ,FLOAT
      ...
      FLOAT ,FLOAT ,FLOAT ,FLOAT
      @ In, fileIn, string, Input file name (absolute path)
      @ Out, data, numpy.ndarray, the loaded data
    """
    # open file
    myFile.open(mode='rb')
    # read the field names
    head = myFile.readline().decode()
    self.allFieldNames = head.split(',')
    for index in range(len(self.allFieldNames)): self.allFieldNames[index] = self.allFieldNames[index].strip()
    # load the table data (from the csv file) into a numpy nd array
    data = np.loadtxt(myFile,dtype='float',delimiter=',',ndmin=2,skiprows=1)
    # close file
    myFile.close()
    return data

  def getFieldNames(self):
    """
      Function to get actual field names (desired output parameter keywords)
      @ In, None
      @ Out, fieldNames, list,  field names' list
    """
    return self.fieldNames

  def getAllFieldNames(self):
    """
      Function to get all field names found in the csv file
      @ In, None
      @ Out, allFieldNames, list, list of field names (headers)
    """
    return self.allFieldNames

  def csvLoadData(self,fileIn,options):
    """
      General interface function to call the private methods for loading the different dataObjects!
      @ In, fileIn, string, csv file name
      @ In, options, dict, dictionary of options
    """
    if   options['type'] == 'Point'   : return self.__csvLoaderForPoint(fileIn[0],options)
    elif options['type'] == 'PointSet': return self.__csvLoaderForPointSet(fileIn,options)
    elif options['type'] == 'History' : return self.__csvLoaderForHistory(fileIn[0],options)
    elif options['type'] == 'HistorySet':
      listhistIn  = {}
      listhistOut = {}
      for index in xrange(len(fileIn)):
        tupleVar = self.__csvLoaderForHistory(fileIn[index],options)
        # dictionary of dictionary key = i => ith history ParameterValues dictionary
        listhistIn[index]  = tupleVar[0]
        listhistOut[index] = tupleVar[1]
        del tupleVar
      return(listhistIn,listhistOut)
    else:
      self.raiseAnError(IOError,'Type ' + options['type'] + 'unknown')

  def __csvLoaderForPoint(self,fileIn,options):
    """
      loader for point data type
      @ In, fileIn, string, file name
      @ In, options, dict, dictionary of options:
            outputPivotVal, output value at which the outputs need to be collected
            inParam, input Parameters
            outParam, output Parameters
            inputRow, outputPivotVal-step from which the input parameters need to be taken
            sampledVars, optional, dictionary of input parameters. The code is going to
                                 look for the inParams in the CSV, if it does not find it
                                 it will try to get the values from this dictionary (if present)
              <inputRow>
              <outputRow>
              <operator>
              <pivotParameter>
              <outputPivotValue>
              <outputPivotValue>
              <inputPivotValue>
      @ Out, (inDict,outDict), tuple, the tuple containing the input and output dictionaries
    """
    inParam, outParam, inputRow, outputRow                 = options['inParam'], options['outParam'], copy.deepcopy(options.get('inputRow',None)), copy.deepcopy(options.get('outputRow',None))
    sampledVars, inputPivotVal, outputPivotVal, operator   = options.get('SampledVars',None), options.get('inputPivotValue',None), options.get('outputPivotValue',None), options.get('operator',None)
    pivotParameter                                         = options.get('pivotParameter',None)

    if 'all' in outParam: self.allOutParam = True
    else                : self.allOutParam = False

    if outputPivotVal != None:
      if 'end' in outputPivotVal: outputPivotValEnd = True
      else:
        outputPivotValEnd, outputPivotVal = False,  float(outputPivotVal)
    else: outputPivotValEnd = True
    if inputRow == None and inputPivotVal == None: inputRow = 0
    if inputRow != None :
      inputRow = int(inputRow)
      if inputRow  > 0: inputRow  -= 1
    if outputRow != None:
      outputRow = int(outputRow)
      if outputRow > 0: outputRow -= 1
    inDict, outDict = {}, {}

    #load the data into the numpy array
    data = self.loadCsvFile(fileIn)
    if pivotParameter != None:
      pivotIndex = self.allFieldNames.index(pivotParameter) if pivotParameter in self.allFieldNames else None
      if pivotIndex == None: self.raiseAnError(IOError,'pivotParameter ' +pivotParameter+' has not been found in file '+ str(fileIn) + '!')
    else:
      pivotIndex = self.allFieldNames.index("time") if "time" in self.allFieldNames else None
      # if None...default is 0
      if pivotIndex == None: pivotIndex = 0
    if inputRow > data[:,0].size-1  and inputRow != -1: self.raiseAnError(IOError,'inputRow is greater than number of actual rows in file '+ str(fileIn) + '!')

    if(self.allOutParam): self.fieldNames = self.allFieldNames
    else: self.fieldNames = outParam

    #fill input param dictionary
    for key in inParam:
      ix = self.allFieldNames.index(key) if key in self.allFieldNames else None
      if ix != None:
        if inputPivotVal != None:
          if float(inputPivotVal) > np.max(data[:,pivotIndex]) or float(inputPivotVal) < np.min(data[:,pivotIndex]): self.raiseAnError(IOError,'inputPivotVal is out of the min and max for input  ' + key+' in file '+ str(fileIn) + '!')
          inDict[key] = np.atleast_1d(np.array(interp1d(data[:,pivotIndex], data[:,ix], kind='linear')(float(inputPivotVal))))
        else: inDict[key] = np.atleast_1d(np.array(data[inputRow,ix]))
      else:
        if sampledVars != None:
          if key in sampledVars.keys(): inDict[key], ix = copy.deepcopy(np.atleast_1d(np.array(sampledVars[key]))), 0
      if ix == None and key != 'InputPlaceHolder': self.raiseAnError(IOError,"the parameter " + key + " has not been found")
    # outputPivotVal end case
    if outputPivotValEnd:
      lastRow = data[:,0].size - 1
      if self.allOutParam:
        for key in self.allFieldNames:
          outDict[key] = np.atleast_1d(np.array(data[lastRow,self.allFieldNames.index(key)]))
      else:
        for key in outParam:
          if key in self.allFieldNames:
            outDict[key] = np.atleast_1d(np.array(data[lastRow,self.allFieldNames.index(key)]))
          else: self.raiseAnError(IOError,"the parameter " + str(key) + " has not been found")
    elif outputRow != None:
      if outputRow > data[:,0].size-1  and outputRow != -1: self.raiseAnError(IOError,'outputRow is greater than number of actual rows in file '+ str(fileIn) + '!')
      if self.allOutParam:
        for key in self.allFieldNames:
          outDict[key] = np.atleast_1d(np.array(data[outputRow,self.allFieldNames.index(key)]))
      else:
        for key in outParam:
          if key in self.allFieldNames:
            outDict[key] = np.atleast_1d(np.array(data[outputRow,self.allFieldNames.index(key)]))
          else: self.raiseAnError(IOError,"the parameter " + str(key) + " has not been found")
    elif operator != None:
      if operator not in ['max','min','average']: self.raiseAnError(IOError,'operator unknown. Available are min,max,average')
      if self.allOutParam:
        for key in self.allFieldNames:
          if operator == 'max'    : outDict[key] = np.atleast_1d(np.array(np.max(data[:,self.allFieldNames.index(key)])))
          if operator == 'min'    : outDict[key] = np.atleast_1d(np.array(np.min(data[:,self.allFieldNames.index(key)])))
          if operator == 'average': outDict[key] = np.atleast_1d(np.array(np.average(data[:,self.allFieldNames.index(key)])))
      else:
        for key in outParam:
          if key in self.allFieldNames:
            if operator == 'max'    : outDict[key] = np.atleast_1d(np.array(np.max(data[:,self.allFieldNames.index(key)])))
            if operator == 'min'    : outDict[key] = np.atleast_1d(np.array(np.min(data[:,self.allFieldNames.index(key)])))
            if operator == 'average': outDict[key] = np.atleast_1d(np.array(np.average(data[:,self.allFieldNames.index(key)])))
          else: self.raiseAnError(IOError,"the parameter " + str(key) + " has not been found")
    else:
      if self.allOutParam:
        for key in self.allFieldNames:
          outDict[key] = np.atleast_1d(np.array(interp1d(data[:,pivotIndex], data[:,self.allFieldNames.index(key)], kind='linear')(outputPivotVal)))
      else:
        for key in outParam:
          if key in self.allFieldNames: outDict[key] = np.atleast_1d(np.array(interp1d(data[:,pivotIndex], data[:,self.allFieldNames.index(key)], kind='linear')(outputPivotVal)))
          else                          : self.raiseAnError(IOError,"the parameter " + key + " has not been found")
    return (inDict,outDict)

  def __csvLoaderForPointSet(self,filesIn,options):
    """
      loader for outputPivotVal point set data type
      @ In, filesIn, string, file name
      @ In, options, dict, dictionary of options:
          outputPivotVal, outputPivotVal
          inParam, input Parameters
          outParam, output Parameters
          inputRow, outputPivotVal-step from which the input parameters need to be taken
          sampledVars, optional, dictionary of input parameters. The code is going to
                                 look for the inParams in the CSV, if it does not find it
                                 it will try to get the values from this dictionary (if present)
              <inputRow>
              <outputRow>
              <operator>
              <pivotParameter>
              <outputPivotValue>
              <outputPivotValue>
              <inputPivotValue>
      @ Out, (inDict,outDict), tuple, the tuple containing the input and output dictionaries
    """
    inParam, outParam, inputRow, outputRow                 = options['inParam'], options['outParam'], copy.deepcopy(options.get('inputRow',None)), copy.deepcopy(options.get('outputRow',None))
    sampledVars, inputPivotVal, outputPivotVal, operator   = options.get('SampledVars',None), options.get('inputPivotValue',None), options.get('outputPivotValue',None), options.get('operator',None)
    pivotParameter                                         = options.get('pivotParameter',None)

    if 'all' in outParam: self.allOutParam = True
    else                : self.allOutParam = False

    if outputPivotVal != None:
      if 'end' in outputPivotVal: outputPivotValEnd = True
      else:
        outputPivotValEnd, outputPivotVal = False,  float(outputPivotVal)
    else: outputPivotValEnd = True
    if inputRow == None and inputPivotVal == None: inputRow = 0
    if inputRow == None and inputPivotVal == None: inputRow = 0
    if inputRow != None :
      inputRow = int(inputRow)
      if inputRow  > 0: inputRow  -= 1
    if outputRow != None:
      outputRow = int(outputRow)
      if outputRow > 0: outputRow -= 1
    inDict, outDict = {}, {}

    for i in range(len(filesIn)):
      #load the data into the numpy array
      data = self.loadCsvFile(filesIn[i])
      if pivotParameter != None:
        pivotIndex = self.allFieldNames.index(pivotParameter) if pivotParameter in self.allFieldNames else None
        if pivotIndex == None: self.raiseAnError(IOError,'pivotParameter ' +pivotParameter+' has not been found in file '+ str(filesIn[i]) + '!')
      else:
        pivotIndex = self.allFieldNames.index("time") if "time" in self.allFieldNames else None
        # if None...default is 0
        if pivotIndex == None: pivotIndex = 0
      if inputRow > data[:,0].size-1  and inputRow != -1: self.raiseAnError(IOError,'inputRow is greater than number of actual rows in file '+ str(filesIn[i]) + '!')

      if i == 0:
        if(self.allOutParam): self.fieldNames = self.allFieldNames
        else: self.fieldNames = outParam
      #fill input param dictionary
      for key in inParam:
        if i == 0: inDict[key] = np.zeros(len(filesIn))
        ix = self.allFieldNames.index(key) if key in self.allFieldNames else None
        if ix != None:
          if inputPivotVal != None:
            if float(inputPivotVal) > np.max(data[:,pivotIndex]) or float(inputPivotVal) < np.min(data[:,pivotIndex]): self.raiseAnError(IOError,'inputPivotVal is out of the min and max for input  ' + key+' in file '+ str(filesIn[i]) + '!')
            inDict[key][i] = interp1d(data[:,pivotIndex], data[:,ix], kind='linear')(float(inputPivotVal))
          else: inDict[key][i] = data[inputRow,ix]
        else:
          if sampledVars != None:
            if key in sampledVars.keys(): inDict[key][i], ix = copy.deepcopy(sampledVars[key]), 0
        if ix == None and key != 'InputPlaceHolder': self.raiseAnError(IOError,"the parameter " + key + " has not been found")
      # outputPivotVal end case
      if outputPivotValEnd:
        lastRow = data[:,0].size - 1
        if self.allOutParam:
          for key in self.allFieldNames:
            if i == 0: outDict[key] = np.zeros(len(filesIn))
            outDict[key][i] = data[lastRow,self.allFieldNames.index(key)]
        else:
          for key in outParam:
            if key in self.allFieldNames:
              if i == 0: outDict[key] = np.zeros(len(filesIn))
              outDict[key][i] = data[lastRow,self.allFieldNames.index(key)]
            else: self.raiseAnError(IOError,"the parameter " + str(key) + " has not been found")
      elif outputRow != None:
        if outputRow > data[:,0].size-1  and outputRow != -1: self.raiseAnError(IOError,'outputRow is greater than number of actual rows in file '+ str(filesIn[i]) + '!')
        if self.allOutParam:
          for key in self.allFieldNames:
            if i == 0: outDict[key] = np.zeros(len(filesIn))
            outDict[key][i] = data[outputRow,self.allFieldNames.index(key)]
        else:
          for key in outParam:
            if key in self.allFieldNames:
              if i == 0: outDict[key] = np.zeros(len(filesIn))
              outDict[key][i] = data[outputRow,self.allFieldNames.index(key)]
            else: self.raiseAnError(IOError,"the parameter " + str(key) + " has not been found")
      elif operator != None:
        if operator not in ['max','min','average']: self.raiseAnError(IOError,'operator unknown. Available are min,max,average')
        if self.allOutParam:
          for key in self.allFieldNames:
            if i == 0: outDict[key] = np.zeros(len(filesIn))
            if operator == 'max'    : outDict[key][i] = np.max(data[:,self.allFieldNames.index(key)])
            if operator == 'min'    : outDict[key][i] = np.min(data[:,self.allFieldNames.index(key)])
            if operator == 'average': outDict[key][i] = np.average(data[:,self.allFieldNames.index(key)])
        else:
          for key in outParam:
            if key in self.allFieldNames:
              if i == 0: outDict[key] = np.zeros(len(filesIn))
              if operator == 'max'    : outDict[key][i] = np.max(data[:,self.allFieldNames.index(key)])
              if operator == 'min'    : outDict[key][i] = np.min(data[:,self.allFieldNames.index(key)])
              if operator == 'average': outDict[key][i] = np.average(data[:,self.allFieldNames.index(key)])
            else: self.raiseAnError(IOError,"the parameter " + str(key) + " has not been found")
      else:
        if self.allOutParam:
          for key in self.allFieldNames:
            if i == 0: outDict[key] = np.zeros(len(filesIn))
            outDict[key][i] = interp1d(data[:,pivotIndex], data[:,self.allFieldNames.index(key)], kind='linear')(outputPivotVal)
        else:
          for key in outParam:
            if i == 0: outDict[key] = np.zeros(len(filesIn))
            if key in self.allFieldNames: outDict[key][i] = interp1d(data[:,pivotIndex], data[:,self.allFieldNames.index(key)], kind='linear')(outputPivotVal)
            else                          : self.raiseAnError(IOError,"the parameter " + key + " has not been found")
      del data
    return (inDict,outDict)

  def __csvLoaderForHistory(self,fileIn,options):
    """
      loader for history data type
      @ In, fileIn, string, file name
      @ In, options, dict, dictionary of options:
          outputPivotVal, outputPivotVal
          inParam, input Parameters
          outParam, output Parameters
          inputRow, outputPivotVal-step from which the input parameters need to be taken
          sampledVars, optional, dictionary of input parameters. The code is going to
                                 look for the inParams in the CSV, if it does not find it
                                 it will try to get the values from this dictionary (if present)
              <inputRow>
              <pivotParameter>
              <outputPivotValue>
              <outputPivotValue>
              <inputPivotValue>
      @ Out, (inDict,outDict), tuple, the tuple containing the input and output dictionaries
    """
    inParam, outParam, inputRow                 = options['inParam'], options['outParam'], copy.deepcopy(options.get('inputRow',None))
    sampledVars, inputPivotVal, outputPivotVal  = options.get('SampledVars',None), options.get('inputPivotValue',None), options.get('outputPivotValue',None)
    pivotParameter                              = options.get('pivotParameter',None)
    #load the data into the numpy array
    data = self.loadCsvFile(fileIn)
    if 'all' in outParam: self.allOutParam = True
    else                : self.allOutParam = False
    if pivotParameter != None:
      pivotIndex = self.allFieldNames.index(pivotParameter) if pivotParameter in self.allFieldNames else None
      if pivotIndex == None: self.raiseAnError(IOError,'pivotParameter ' +pivotParameter+' has not been found in file '+ str(fileIn) + '!')
    else:
      pivotIndex = self.allFieldNames.index("time") if "time" in self.allFieldNames else None
      # if None...default is 0
      if pivotIndex == None: pivotIndex = 0

    if outputPivotVal != None:
      if 'all' in outputPivotVal: outputPivotValAll = True
      else:
        outputPivotValAll, outputPivotVal = False,  [float(x) for x in outputPivotVal.split()]
    else: outputPivotValAll = True
    if inputRow == None and inputPivotVal == None: inputRow = 0
    if inputRow == None and inputPivotVal == None: inputRow = 0
    if inputRow != None :
      inputRow = int(inputRow)
      if inputRow  > 0: inputRow  -= 1
    if inputRow > data[:,0].size-1  and inputRow != -1: self.raiseAnError(IOError,'inputRow is greater than number of actual rows in file '+ str(fileIn) + '!')
    inDict, outDict = {}, {}
    self.fieldNames = self.allFieldNames if self.allOutParam else outParam

    #fill input param dictionary
    for key in inParam:
      inDict[key] = np.zeros(1)
      ix = self.allFieldNames.index(key) if key in self.allFieldNames else None
      if ix != None:
        if inputPivotVal != None:
          if float(inputPivotVal) > np.max(data[:,pivotIndex]) or float(inputPivotVal) < np.min(data[:,pivotIndex]): self.raiseAnError(IOError,'inputPivotVal is out of the min and max for input  ' + key+' in file '+ str(fileIn) + '!')
          inDict[key] = np.atleast_1d(np.array(interp1d(data[:,pivotIndex], data[:,ix], kind='linear')(float(inputPivotVal))))
        else: inDict[key] = np.atleast_1d(np.array(data[inputRow,ix]))
      else:
        if sampledVars != None:
          if key in sampledVars.keys(): inDict[key], ix = copy.deepcopy(np.atleast_1d(sampledVars[key])), 0
      if ix == None and key != 'InputPlaceHolder': self.raiseAnError(IOError,"the parameter " + key + " has not been found")

    # outputPivotVal all case
    if outputPivotValAll:
      if self.allOutParam:
        for key in self.allFieldNames:
          outDict[key] = data[:,self.allFieldNames.index(key)]
      else:
        for key in outParam:
          if key in self.allFieldNames:
            outDict[key] = data[:,self.allFieldNames.index(key)]
          else:
            self.raiseAnError(IOError,"the parameter " + key + " has not been found")
    else:
      # pivot value
      if self.allOutParam:
        for key in self.allFieldNames: outDict[key] = np.atleast_1d(np.array(interp1d(data[:,pivotIndex], data[:,self.allFieldNames.index(key)], kind='linear')(outputPivotVal)))
      else:
        for key in outParam:
          if key in self.allFieldNames: outDict[key] = np.atleast_1d(np.array(interp1d(data[:,pivotIndex], data[:,self.allFieldNames.index(key)], kind='linear')(outputPivotVal)))
          else                          : self.raiseAnError(IOError,"the parameter " + key + " has not been found")
    return (inDict,outDict)
