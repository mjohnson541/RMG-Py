    #!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 7 11:38:25 2017

@author: mattjohnson
"""

"""
QueueGenerator script

syntax:  python QueueGenerator.py rmgDir runDir outDir

This script uses RMG from rmgDir, and the input file and chem_annotated.inp file it finds in runDir
to generate a json format file named 'queue.json' in outDir

It does this in an infinite while loop such that it will continually grab the input file and 
the latest chem_annotated.inp file from runDir and replace the old 'queue.json' file
"""

import os
import sys
import json
import re

#rmgDir = '/Users/mattjohnson/RMGCODE/RMG-Py' #SET THIS the directory of RMG-Py
#runDir = '/Users/mattjohnson/RMGCODE/RMG-Py/examples/rmg/syngas' #SET THIS the run directory of the RMG job
#outDir = '/Users/mattjohnson/Desktop' 

rmgDir = sys.argv[1] #rmg directory
runDir = sys.argv[2] #the run directory of the RMG job
outDir = sys.argv[3] #directory the queue.json will be made in
    
os.chdir(rmgDir)

import rmgpy
from rmgpy.tools.sensitivity import runSensitivity
from rmgpy.tools.uncertainty import Uncertainty, ThermoParameterUncertainty, KineticParameterUncertainty
from rmgpy.chemkin import *
from rmgpy.tools.canteraModel import *
from rmgpy.tools.plot import parseCSVData


spcs_regex = '[d][G][\[][#\)\(\w]*[\]]'
spcre = re.compile(spcs_regex)

def getThermoLabel(d):
    """
    retrives the species name associated with the data entry
    """
    x = spcre.findall(d.label)
    if len(x) > 1:
        logging.warn('thermo label retrival from {0} may have been unsafe'.format(d.label))
    return x[0][3:-1]

def reduceData(d):
    """
    reduces the time dependent sensitivities down to one number
    """
    return max(abs(d.data))

spcs_label_regex = '[l][a][b][e][l][\n\w]*[=][\n\w]*[\"\'][\(\)#\w]*[\"\']'
splb = re.compile(spcs_label_regex)

def makeSensInputFile(inputFile):
    """
    reads the original input file and generates a new input file
    that does sensitivity analysis on the first species listed in the file
    """
    #get input file name
    inputPath,inputName = os.path.split(inputFile)
    inputName = inputFile.split('/')[-1][0:-3]
    #get the original input file text
    f = open(inputFile,'rb')
    txt = f.readlines()
    wtxt = ''
    for s in txt:
        wtxt += s
        
    #get the first species name
    spcs = splb.findall(wtxt)
    assert len(spcs) > 0, wtxt
    spc = spcs[0]
    if '\"' in spc:
        ind1 = spc.find('\"')
        ind2 = spc.rfind('\"')
    elif '\'' in spc:
        ind1 = spc.find('\'')
        ind2 = spc.rfind('\'')
    spcstr = spc[ind1+1:ind2]
    line = 'sensitivity=['+'\''+spcstr+'\''+'],\n'+'sensitivityThreshold=0.0,\n'
    
    #find reactor section
    rxnind = wtxt.find('Reactor(')
    rxnind = rxnind+7
    endind = findCloseParen(wtxt,rxnind)
    
    rtxt = wtxt[rxnind+1:endind]
    eq = rtxt.count('=')
    commas = rtxt.count(',')
    parenPairs = rtxt.count(')')+rtxt.count('}')

    if eq == commas-parenPairs:
        newtxt = wtxt[:endind]+line+wtxt[endind:]
    else:
        ind = endind-1
        while not (wtxt[ind].isalpha() or wtxt[ind].isdigit() or wtxt[ind] == ')'):
            ind -= 1
        newtxt = wtxt[:ind+2]+line+wtxt[endind:]
    
    newFName = inputName+'Sens'+'.py'
    f = open(os.path.join(inputPath,newFName),'wb')
    f.write(newtxt)
    f.close()
    
    assert os.path.exists(os.path.join(inputPath,newFName))
    
    return newFName
    
def findCloseParen(wtxt,ind):
    """
    if ind is the index an open paren occurs at within wtxt
    this function will return the close paren that closes
    that open paren
    """
    a = 1#'(' start at the open paren
    b = 0#')'
    while a > b:
        ind += 1
        s = wtxt[ind]
        if s == '(':
            a += 1
        elif s == ')':
            b += 1
    return ind

def getSens(runDir,inputFile):
    """
    calculates sensitivities, returns a list of sensitivity values in 
    species order
    thermo sensitivity values are in 1/(kcal/mol)
    """
    
    chemDir = os.path.join(runDir,'chemkin')
    chemkinFile = os.path.join(chemDir,'chem_annotated.inp')
    spcDict = os.path.join(chemDir,'species_dictionary.txt')
    
    assert os.path.exists(inputFile)
    
    runSensitivity(inputFile,chemkinFile,spcDict)
    
    sensStr = os.path.join('solver','sensitivity_1_SPC_1.csv')
    strInd = sensStr.find('1')
    q = 1
        
    time, dataList = parseCSVData(os.path.join(runDir,'solver','sensitivity_1_SPC_1.csv'))
    
    thermoDataList = [d for d in dataList if 'dG[' in d.label]
    #rxnDataList = list(set(dataList)-set(thermoDataList))
        
    thermoSens = [reduceData(d) for d in thermoDataList]
    
    #all other reactors
    while True:
        q += 1
        newSensStr = sensStr[:strInd]+str(q)+sensStr[strInd+1:]
        try:
            time, dataList = parseCSVData(runDir+newSensStr)
        except:
            break
        thermoDataList = [d for d in dataList if 'dG[' in d.label]
        tempSens = [reduceData(d) for d in thermoDataList]
        for i in xrange(len(thermoSens)):
            if thermoSens[i] < tempSens[i]:
                thermoSens[i] = tempSens[i]
                
    return thermoSens

def getUncertainties(runDir,thermoLibraries,reactionLibraries):
    """
    Calculates uncertainties, returns a list of uncertainty values in 
    species order
    thermo uncertainty values are in kcal/mol
    """
    
    chemDir = os.path.join(runDir,'chemkin')
    chemkinFile = os.path.join(chemDir,'chem_annotated.inp')
    spcDict = os.path.join(chemDir,'species_dictionary.txt')
    
    try:
        os.mkdir('UncertaintyDir')
    except OSError:
        pass
    
    uncertainty = Uncertainty(outputDirectory='UncertaintyDir')
    uncertainty.loadModel(chemkinFile, spcDict)
    uncertainty.loadDatabase(thermoLibraries=thermoLibraries,reactionLibraries=reactionLibraries)
    uncertainty.extractSourcesFromModel()
    uncertainty.compileAllSources()
    uncertainty.assignParameterUncertainties()
    gParamEngine = ThermoParameterUncertainty()
    #kParamEngine = KineticParameterUncertainty()
    
    thermoUnc = uncertainty.thermoInputUncertainties
    return thermoUnc

def getSpcs(runDir):
    """
    retrives the species objects
    """
    chemDir = os.path.join(runDir,'chemkin')
    chemkinFile = os.path.join(chemDir,'chem_annotated.inp')
    spcDict = os.path.join(chemDir,'species_dictionary.txt')
    species,reactions = loadChemkinFile(chemkinFile,spcDict)
    return species

def getLibraries(inputFile):
    """
    retrieves the thermo and kinetics library names from the input file
    """
    f = open(inputFile,'rb')
    txt = f.readlines()
    thermoLibs = []
    kineticsLibs = []
    txt = [line for line in txt]
    for ind,line in enumerate(txt):
        if 'thermoLibraries' in line:
            assert not '#' in line, 'cannot have comments in thermoLibraries line'
            
            evalstr = line.split('=')[1].strip()
            
            thermoLibs = eval(evalstr)[0]
            
            for i in xrange(len(thermoLibs)):
                if type(thermoLibs[i]) == tuple:
                    thermoLibs[i] = thermoLibs[i][0]
            
                
        if 'reactionLibraries' in line:
            assert not '#' in line, 'cannot have comments in reactionLibraries line'
            
            evalstr = line.split('=')[1].strip()
            
            kineticsLibs = eval(evalstr)[0]
            
            for i in xrange(len(kineticsLibs)):
                if type(kineticsLibs[i]) == tuple:
                    kineticsLibs[i] = kineticsLibs[i][0]
                
        if kineticsLibs != [] and thermoLibs != []:
            break
    
    return thermoLibs,kineticsLibs

R = 1.987e-3 #kcal/K

class QueueEntry(object):
    
    def __init__(self,value):
        self.value = value
        
    def __cmp__(self,obj):
        dif = self.value-obj.value
        if dif > 0:
            return 1
        elif dif == 0:
            return 0
        elif dif < 0:
            return -1

import logging

class ThermoQueueEntry(QueueEntry):
    
    def __init__(self,spc,sensitivity,uncertainty):
        self.spc = spc
        self.label = str(spc)
        self.sensitivity = sensitivity
        self.uncertainty = uncertainty
        QueueEntry.__init__(self,sensitivity*np.exp(uncertainty/(R*300.0)))
    
    def __str__(self):
        spc = self.spc
        s = ''
        s += '('
        s += spc.generate_aug_inchi()+', '
        s += spc.toAdjacencyList()+', '
        s += str(spc.thermo)+', '
        s += str(self.uncertainty) + ', '
        s += str(self.value)+', '
        s += 'waiting' +', '
        s += ')'
        return s
    
    def toDict(self):
        smiles = self.spc.molecule[0].toSMILES()
        inchi = self.spc.molecule[0].toInChI()
        extendedinchi = inchi+'/u'+str(self.spc.molecule[0].multiplicity)
            
        return {'name':self.spc.label,
                'ExtendedInchi':extendedinchi,
                'InChi':inchi,
                'multiplicity':self.spc.molecule[0].multiplicity,
                'SMILES': smiles,
                'AdjList':self.spc.toAdjacencyList(),
                'Uncertainty':self.uncertainty,
                'Sensitivity':self.sensitivity,
                'Value':self.value,
                'H298':self.spc.thermo.getEnthalpy(298.0),
                'S298': self.spc.thermo.getEntropy(298.0),
                'Cp300': self.spc.thermo.getHeatCapacity(300.0),
                'Cp500': self.spc.thermo.getHeatCapacity(500.0),
                'Cp1000': self.spc.thermo.getHeatCapacity(1000.0),
                    'Cp1500': self.spc.thermo.getHeatCapacity(1500.0),
                }
        
if __name__ == "__main__":
    
    while True:
        
        inputFile = os.path.join(runDir,'input.py')
        inputFile = os.path.join(runDir,makeSensInputFile(inputFile))
        
        species = getSpcs(runDir)
        thermoLibraries,reactionLibraries = getLibraries(inputFile)
        
        thermoUnc = getUncertainties(runDir,thermoLibraries,reactionLibraries)

        thermoSens = getSens(runDir,inputFile)
        
        thermoUnc = thermoUnc[4:]
        species = species[4:]
                
        queue = []
        
        for i in xrange(len(thermoSens)):
            queue.append(ThermoQueueEntry(species[i],thermoSens[i],thermoUnc[i]))

        
        sortedQueue = sorted(queue)[::-1]
                
        dictQueue = []
        for entry in sortedQueue:
            dictQueue.append(entry.toDict())
            
        #print to json
        fname = 'queue.json'
            
        os.chdir(outDir)
        fid = open(fname,'wb')
            
        json.dump(dictQueue,fid)
            
        fid.close()

