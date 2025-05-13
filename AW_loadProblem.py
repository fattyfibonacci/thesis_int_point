#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load some mat-files into python.
Created on Wed Sep 29 12:42:00 2021

@author: Andreas Wachtel
"""

import scipy.io
def loadProblem( fname, useSparse = False ):
    mat = scipy.io.loadmat( fname )
    if useSparse:
        A = mat.get('Problem')[0,0][2].astype(float) # sparse matrix
    else:
        A = mat.get('Problem')[0,0][2].astype(float).toarray()
    
    b = mat.get('Problem')[0,0][3].astype(float)[:,0]
    #id = mat.get('Problem')[0,0][4]
    aux = mat.get('Problem')[0,0][5]
    c = aux[0,0][0].astype(float)[:,0]
    
    return {'AE':A, 'bE':b, 'c':c}


if __name__ == '__main__':
    H = loadProblem('lp_adlittle.mat')
    print('type(A) : ', type(H['AE']))
    print(H['bE'].shape)
    print(H['c'].shape)
    #print(H)
    print("This file was executed from the command line or an interpreter.")
    
else:
    print("Imported: loadProblem (by AW).")
