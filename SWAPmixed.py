from __future__ import division
from scipy.integrate import ode
from scipy import sparse
from scipy.stats import uniform
from scipy.optimize import *
import numpy as np
import math
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#np.set_printoptions(threshold=sys.maxsize)

def CSWAP(n, rho):
    #print('Number of qubits in test state =')
    #print(n)
    N = 2**n  # number of configurations

    #CONTROL (after Hadamard)
    C = np.zeros([N,1], dtype = int)
    for i in range(N):
        C[i] = 1
    rhoC = C * np.transpose(C)
        
    #SUPERPOSTION
    psi = np.kron(rhoC, rho)
    AB=None
    C=None
    rhoC=None

    # CSWAP : FOR EVERY QUBIT
    M = len(psi)
    MN = np.rint(M / N).astype(int)
    N2 = np.rint(N / 2).astype(int)
    nCSWAP = sparse.identity(M, dtype=int)
    for q in range(1,n+1):              # FOR n QUBITS
        CSWAP = sparse.eye(M, dtype=int, format='dok') #format? #dok/lil?
        Q = np.rint(2**q).astype(int)
        Q2 = np.rint(Q/2).astype(int)
        for s in range(0, M, (Q*MN)):   # STARTS OF NO SWAP - SWAP CYCLES
            i = s                       # set i = start
            for l in range(Q2*MN):      # NO SWAP FOR Q/2 * M/N
                i = i + 1
            for l in range(N2):         # SWAP FOR Q/2 * M/N        
                for k in range(N2):    
                    for j in range(Q2): # NO SWAP FOR Q/2
                        i = i + 1
                    for j in range(Q2): # SWAP FOR Q/2
                        CSWAP[i,i] = 0
                        CSWAP[i,i+(Q2*(N-1))] = 1
                        CSWAP[i+(Q2*(N-1)),i] = 1
                        i = i + 1
                for k in range(N2):    
                    for j in range(Q2): # SWAP FOR Q/2
                        CSWAP[i,i] = 0
                        i = i + 1
                    for j in range(Q2): # NO SWAP FOR Q/2
                        i = i + 1

        nCSWAP = CSWAP*nCSWAP

    CSWAP=None
    psi2 = nCSWAP * psi * np.transpose(nCSWAP)
    nCSWAP=None
    T = None

    # HADAMARD
    # n = 1
    H = np.zeros([2,2], dtype=int)
    H[0,0] = 1
    H[0,1] = 1
    H[1,0] = 1
    H[1,1] = -1
        
    # n
    Hn = H
    for j in range(2, n+1):                # for each n
        H_ = Hn                            # previous Hn
        Nj = np.rint(2**j).astype(int)     # current N
        N_ = np.rint(Nj/2).astype(int)     # last N
        Hn = np.zeros([Nj,Nj], dtype=int)  # new Hn
        for c in range(N_):
            H2 = np.kron(H[0], H_[c])
            for d in range(Nj):
                Hn[c,d] = H2[d]
        for c in range(N_):
            H2 = np.kron(H[1], H_[c])
            for d in range(Nj):
                Hn[c+N_,d] = H2[d]

    H2=None
    
    # as matrix (diagonal)
    diagonals = [0] * ((2*N)-1)
    offset = [0] * ((2*N)-1)

    for i in range((2*N)-1):
        offset[i] = -((N-(i+1))*MN)
        
    for i in range(N):  # lower diagonals (and central)
        diagonals[i] = np.array(Hn[N-(i+1), 0])
        for j in range(i):
            diagonals[i] = np.append(diagonals[i], Hn[N-((i+1)-(j+1)), j+1])

    for i in range(N, (2*N)-1):  # upper diagonals
        k = - i + (2*N) - 2
        diagonals[i] = np.array(Hn[0, N-(k+1)])
        for j in range(k):
            diagonals[i] = np.append(diagonals[i], Hn[j+1, N-((k+1)-(j+1))])

    kr = np.ones(MN, dtype=int)
    for i in range((2*N)-1):
        diagonals[i] = np.kron(diagonals[i], kr)  # M/N of each

    Hn=None
    HC = sparse.diags(diagonals, offsets=offset, dtype = int)

    # APPLY HADAMARD
    psi3 = HC * psi2 * np.transpose(HC)
    psi3 = psi3 * (np.sqrt(1/2))**(2*n) * (np.sqrt(1/2))**(2*n)  # extra for Hadamards
    psi2=None
    HC=None

    # CHECK
    Ps = np.diagonal(psi3)
    print(Ps)
    print('END CHECK: total probability =')
    print(np.sum(Ps))
    
    P = np.zeros(N)
    L = len(Ps)
    part = int(L/N)
    for i in range(N):
        low = i*part
        high = (i+1)*part
        P[i] = np.sum(Ps[low:high])

    # to print each qubit string C
    Cq = np.zeros([N,n], dtype=int)
    i = 0
    for q in range(1, n+1):
        Q = np.rint(2**q).astype(int)
        Q2 = np.rint(Q/2).astype(int)
        i = 0
        qi = n - q
        QI = np.rint(2**qi).astype(int)
        for k in range(QI):
	        for j in range(Q2): 
	            Cq[i,qi] = 0
	            i = i + 1
	        for j in range(Q2):
	            Cq[i,qi] = 1
	            i = i + 1

    print('Probability of measuring in each C:')
    for i in range(N):
        print(i, Cq[i], P[i])  # print array designation, qubit string, and probability
        
    #print("P10 + P11", P[2] + P[3])
    #print(P[8] + P[9] + P[10] + P[11] + P[12] + P[13] + P[14] + P[15])
	#psi3
	#ai bi
	#aj bj
	
    #print(len(psi3))
    X0 = np.asarray([1,0])
    X0 = np.transpose(X0)
    X1 = np.asarray([0,1])
    X1 = np.transpose(X1)
    I1 = np.asarray([1,1])
    I1 = np.transpose(I1)
    
    # i
    if (n==2):
        tracei00 = np.kron(I1,np.kron(I1,np.kron(X0,np.kron(I1,np.kron(X0, I1)))))
        tracei01 = np.kron(I1,np.kron(I1,np.kron(X0,np.kron(I1,np.kron(X1, I1)))))
        tracei10 = np.kron(I1,np.kron(I1,np.kron(X1,np.kron(I1,np.kron(X0, I1)))))
        tracei11 = np.kron(I1,np.kron(I1,np.kron(X1,np.kron(I1,np.kron(X1, I1)))))
    elif (n==3):
        tracei00 = np.kron(I1,np.kron(I1,np.kron(I1, np.kron(X0,np.kron(I1,np.kron(I1, np.kron(X0, np.kron(I1,I1))))))))
        tracei01 = np.kron(I1,np.kron(I1,np.kron(I1, np.kron(X0,np.kron(I1,np.kron(I1, np.kron(X1, np.kron(I1,I1))))))))
        tracei10 = np.kron(I1,np.kron(I1,np.kron(I1, np.kron(X1,np.kron(I1,np.kron(I1, np.kron(X0, np.kron(I1,I1))))))))
        tracei11 = np.kron(I1,np.kron(I1,np.kron(I1, np.kron(X1,np.kron(I1,np.kron(I1, np.kron(X1, np.kron(I1,I1))))))))
    elif (n==4):
        tracei00 = np.kron(I1,np.kron(I1,np.kron(I1,np.kron(I1, np.kron(X0,np.kron(I1,np.kron(I1,np.kron(I1, np.kron(X0, np.kron(I1,np.kron(I1,I1)))))))))))
        tracei01 = np.kron(I1,np.kron(I1,np.kron(I1,np.kron(I1, np.kron(X0,np.kron(I1,np.kron(I1,np.kron(I1, np.kron(X1, np.kron(I1,np.kron(I1,I1)))))))))))
        tracei10 = np.kron(I1,np.kron(I1,np.kron(I1,np.kron(I1, np.kron(X1,np.kron(I1,np.kron(I1,np.kron(I1, np.kron(X0, np.kron(I1,np.kron(I1,I1)))))))))))
        tracei11 = np.kron(I1,np.kron(I1,np.kron(I1,np.kron(I1, np.kron(X1,np.kron(I1,np.kron(I1,np.kron(I1, np.kron(X1, np.kron(I1,np.kron(I1,I1)))))))))))
        
    psiABi00 = (np.transpose(tracei00) * psi3 * tracei00)
    psiABi01 = (np.transpose(tracei01) * psi3 * tracei01)
    psiABi10 = (np.transpose(tracei10) * psi3 * tracei10)
    psiABi11 = (np.transpose(tracei11) * psi3 * tracei11)
    psiABi0110 = (np.transpose(tracei01) * psi3 * tracei10)
    psiABi1001 = (np.transpose(tracei10) * psi3 * tracei01)
    psiABiBell = 1/2 * (psiABi01 + psiABi0110 + psiABi1001 + psiABi10)
    PiBell = np.sum(np.diagonal(psiABiBell))
    #print(PiBell)
    
    return(P)
    
# CREATE STATE(S)
def states(n, theta, phi):
	N = 2**n             # number of configurations
	A = np.zeros([N,1])  # test state A
	B = np.zeros([N,1])  # test state B
    # USE THETA FOR A=B FIT OR CONSTANT A WITH B FIT
    # USE THETA AND PHI FOR UNIQUELY VARYING A AND B FITS

    ################# INPUT: FIRST STATE
    #3-qubit Bell
	#A[0] = 1/np.sqrt(2)
	#A[3] = 1/np.sqrt(2)
	#B[0] = np.cos(theta)
	#B[3] = np.sin(theta)
	
	#A[0] = 1/np.sqrt(2)
	#A[3] = 1/np.sqrt(2)
	
	#W3 state
	#a = 1/np.sqrt(3)
	#A[1] = a
	#A[2] = a
	#A[4] = a
	
	# W
	for v in range(n):
		w = 2**v
		A[w] = 1/np.sqrt(n)
	rhoW = np.transpose(A) * A
	
	#almost W
	#a = np.cos(theta) / np.sqrt(3)
	#B[1] = a
	#B[2] = a
	#B[4] = a
	#B[3] = np.sin(theta)
	
	# GHZ
	A = np.zeros([N,1])
	A[0] = 1/np.sqrt(2)
	A[N-1] = 1/np.sqrt(2)
	rhoGHZ = np.transpose(A) * A
	
	# 22
	#if n==4:
	#	A[0] = np.cos(theta)*np.cos(phi)
	#	A[3] = np.cos(theta)*np.sin(phi)
	#	A[12] = np.sin(theta)*np.cos(phi)
	#	A[N-1] = np.sin(theta)*np.sin(phi)
		#A = A/np.sqrt(np.cos(theta + np.pi/4)**4 + 2*np.cos(theta + np.pi/4)**2*np.sin(theta + np.pi/4)**2 + np.sin(theta + np.pi/4)**4) 
	
	# GHZ flip
	A = np.zeros([N,1])
	if n==3:
		A[1] = np.sin(theta)
		A[6] = np.cos(theta)
	
	# general GHZ
	#A = np.zeros([N,1])
	#x = 1/np.sqrt(3)
	#A[0] = x
	#A[N-1] = np.sqrt(1-x**2)
	#rhoGHZ = np.transpose(A) * A
	
	#almost GHZ
	#A[0] = 1/2
	#A[7] = 1/2
	#A[1] = 1/np.sqrt(2)
	#B[0] = np.cos(theta) / np.sqrt(2)
	#B[7] = B[0]
	#B[1] = np.sin(theta

	#a = np.sqrt(2)
	#A[1] = np.sin(theta) / a
	#A[2] = np.cos(theta) /a
	#A[5] = A[2]
	#A[6] = A[1]
	
	# combo
	#A[0] = 1/2 #np.sin(theta) /np.sqrt(2)
	#A[N-1] = A[0]
	#for d in range(n):
	#	A[2**d] = 1/np.sqrt(2*n)#np.cos(theta) / np.sqrt(n)
		
	# cluster state
	#A = np.ones([N,1])
	#A[3] = -1
	#A[6] = -1
	#A[11] = -1
	#A[12] = -1
	#A[13] = -1
	#A[15] = -1
	#A = A * 7 * np.sqrt(465) / 600
	#A[0] = 1/4 - 1/40
	
	# magic states
	A = np.zeros([N,1])
	A[0] = 1/2
	A[5] = 1/2
	A[10] = 1/2
	A[15] = 1/2
	
	# combo B
	#B[0] = np.cos(theta + np.pi/4) /np.sqrt(2)
	#B[N-1] = B[0]
	#for d in range(n):
	#	B[2**d] = np.sin(theta + np.pi/4) / np.sqrt(n)
	    
    #DENSITY MATRICES
	rhoA = np.transpose(A) * A
	#rhoB = np.transpose(B) * B
	A=None
	B=None
    
	#rhoA = np.zeros([N, N])
	#rhoB = np.zeros([N, N])
	
	# max mixed
	#rhoA = np.identity(N, dtype=int)/N
	
	# max mixed 2 Bells
	Bell1 = np.zeros([4,1])
	Bell1[0] = np.sqrt(1/2)
	Bell1[3] = np.sqrt(1/2)
	Bell2 = np.zeros([4,1])
	Bell2[0] = np.sqrt(1/2)
	Bell2[3] = - np.sqrt(1/2)
	Bell3 = np.zeros([4,1])
	Bell3[1] = np.sqrt(1/2)
	Bell3[2] = np.sqrt(1/2)
	Bell4 = np.zeros([4,1])
	Bell4[1] = np.sqrt(1/2)
	Bell4[2] = - np.sqrt(1/2)
	#rhoA = (np.kron((np.transpose(Bell1) * Bell1), (np.transpose(Bell1) * Bell1)) + np.kron((np.transpose(Bell2) * Bell2), (np.transpose(Bell2) * Bell2)) + np.kron((np.transpose(Bell3) * Bell3), (np.transpose(Bell3) * Bell3)) + np.kron((np.transpose(Bell4) * Bell4), (np.transpose(Bell4) * Bell4)))/4
	
	# probability max mixed
	#rhoA = ((1-theta) * rhoGHZ) + (theta * np.identity(N, dtype=int)/N)
	#rhoA = ((1-theta) * rhoW) + (theta * np.identity(N, dtype=int)/N)
	#p=phi
	#rhoA = ((1-p) * rhoGHZ) + (p * np.identity(N, dtype=int)/N)
	
	# conc=1/2
	#rA = [[1, np.sqrt(3), np.sqrt(3), 1], [np.sqrt(3), 3, 3, np.sqrt(3)], [np.sqrt(3), 3, 3, np.sqrt(3)], [1, np.sqrt(3), np.sqrt(3), 1]]
	#rhoA1 = 1/8 * np.asarray(rA)
	
	# conc = 1/sqrt(2)
	#rA = [[2, np.sqrt(2), np.sqrt(2), 2], [np.sqrt(2), 2, 2, np.sqrt(2)], [np.sqrt(2), 2, 2, np.sqrt(2)], [2, np.sqrt(2), np.sqrt(2), 2]]
	#rhoA = 1/8 * np.asarray(rA)
	
	# 2-qubit delta=pi/4
	#x = 2*np.cos(theta)*np.sin(theta)
	#rA = [[1, x, x, 1], [x, 1, 1, x], [x, 1, 1, x], [1, x, x, 1]]
	#rhoA = 1/4 * np.asarray(rA)
	
	# 2-qubit, pure=sin(pi/8) 00 + cos(pi/8) 11
	#a = np.pi/8
	#x = np.cos(a)*np.sin(a)
	#X = np.cos(a)**2 * np.cos(theta)**2 + np.sin(a)**2 * np.sin(theta)**2
	#Y = np.sin(a)**2 * np.cos(theta)**2 + np.cos(a)**2 * np.sin(theta)**2
	#rA = [[X, x, x, X], [x, Y, Y, x], [x, Y, Y, x], [X, x, x, X]]
	#rhoA = 1/2 * np.asarray(rA)
    
    # GHZ STATE
    #rhoA[0][0] = 0.5
    #rhoA[N-1][N-1] = 0.5
    #rhoA[0][N-1] = np.cos(theta) * np.sin(theta)
    #rhoA[N-1][0] = np.cos(theta) * np.sin(theta)
    
    # W STATE
    #for v in range(n):
     #   w = 2**v
      #  rhoA[w][w] = 1/n
       # cross = (((2 / np.sqrt(n-1)) * np.sin(theta) * np.cos(theta)) + (((n-2)/(n-1)) * (np.sin(theta))**2))/n
        #for s in range(n):
         #   u = 2**s
          #  if u != w:
           #     rhoA[w][u] = cross
           
	#BOTH
	#cs = (2/np.sqrt(2*n)) * np.cos(theta) * np.sin(theta)
	#rhoA[0][2**(n-1)] = cs
	#rhoA[0][2**n-1] = cs
	#rhoA[2**(n-1)][0] = cs
	#rhoA[2**n-1][0] = cs
	#rhoA[0][0] = 1/n
	#rhoA[2**n-1][2**n-1] = 1/2
	#rhoA[2**(n-1)][2**(n-1)] = 1/2
	#rhoA[2**n-1][2**(n-1)] = 1/2
	#rhoA[2**(n-1)][2**n-1] = 1/2
	#for i in range(n-1):
		#rhoA[2**i][2**(n-1)] = cs
		#rhoA[0][2**i] = 1/n
		#rhoA[2**i][0] = 1/n
		#rhoA[2**i][2**n-1] = cs
		#rhoA[2**(n-1)][2**i] = cs
		#rhoA[2**n-1][2**i] = cs
		#for j in range(n-1):
			#rhoA[2**i][2**j] =1/n
	#half = [1/2] * len(rhoA)
	#rhoA = np.multiply(half, rhoA)
	
	# COMBO
	#cross = np.sin(theta) * np.sin(phi) * np.cos(theta) * np.cos(phi)
	#rhoA = np.full([N, N], cross)
	#r03 = (np.sin(phi))**2 /2
	#r12 = (np.cos(phi))**2 /2
	#rhoA[0][0] = r03
	#rhoA[1][1] = r12
	#rhoA[2][2] = r12
	#rhoA[3][3] = r03
	#rhoA[0][3] = r03
	#rhoA[3][0] = r03
	#rhoA[1][2] = r12
	#rhoA[2][1] = r12
	
	# C2 = |cos(2 theta)|
	#a = (np.cos(phi)**2 * np.cos(theta)**2 + np.sin(phi)**2 * np.sin(theta)**2) /2
	#b  = np.cos(theta) * np.sin(theta) / 2
	#c = (np.cos(phi)**2 * np.sin(theta)**2 + np.sin(phi)**2 * np.cos(theta)**2) /2
	#for k in [0, 3]:
	#	for g in [0, 3]:
	#		rhoA[k, g] = a
	#	for g in [1, 2]:
	#		rhoA[k, g] = b
	#for k in [1, 2]:
	#	for g in [0, 3]:
	#		rhoA[k, g] = b
	#	for g in [1, 2]:
	#		rhoA[k, g] = c
	
	# unbalanaced GHZ
	#a = np.cos(theta)
	#b = np.sin(theta)
	#s = np.sin(phi)
	#c = np.cos(phi)
	#rhoA[0,0] = c**2 * a**2 + s**2 * b**2
	#rhoA[0,7] = 2 * (a*b)
	#rhoA[7,0] = 2 * (a*b)
	#rhoA[7,7] = c**2 * b**2 + s**2 * a**2
	
    ################# INPUT: SECOND STATE
	rhoB = rhoA
	#q = theta
	#rhoB = ((1-q) * rhoGHZ) + (q * np.identity(N, dtype=int)/N)
    
    # conc=1/2
	#rA1 = [[1, np.sqrt(3), np.sqrt(3), 1], [np.sqrt(3), 3, 3, np.sqrt(3)], [np.sqrt(3), 3, 3, np.sqrt(3)], [1, np.sqrt(3), np.sqrt(3), 1]]
	#rhoA1 = 1/8 * np.asarray(rA)
	#rB = [[np.sin(theta)**2, 0, 0, np.sin(theta)**2], [0, -np.sin(theta)**2, -np.sin(theta)**2, 0], [0, -np.sin(theta)**2, -np.sin(theta)**2, 0], [np.sin(theta)**2, 0, 0, np.sin(theta)**2]]
	#rB = 1/4 * np.asarray(rB)
	#rhoB = rhoA1 + rB
	
	#  2-qubit, delta=pi/8
	#x = np.cos(theta)*np.sin(theta)
	#X = np.cos(theta)**2 * np.cos(np.pi/8)**2 + np.sin(theta)**2 * np.sin(np.pi/8)**2
	#Y = np.sin(theta)**2 * np.cos(np.pi/8)**2 + np.cos(theta)**2 * np.sin(np.pi/8)**2
	#rB = [[X, x, x, X], [x, Y, Y, x], [x, Y, Y, x], [X, x, x, X]]
	#rhoB = 1/2 * np.asarray(rB)
	
	#a = np.pi/3
	#x = np.cos(a)*np.sin(a)
	#X = np.cos(a)**2 * np.cos(phi)**2 + np.sin(a)**2 * np.sin(phi)**2
	#Y = np.sin(a)**2 * np.cos(phi)**2 + np.cos(a)**2 * np.sin(phi)**2
	#rB = [[X, x, x, X], [x, Y, Y, x], [x, Y, Y, x], [X, x, x, X]]
	#rhoB = 1/2 * np.asarray(rA)
    
    #GHZ
	#rhoB[0][0] = (np.cos(phi))**2
	#rhoB[N-1][N-1] = (np.sin(phi))**2
	#rhoB[0][N-1] = np.cos(theta) * np.sin(theta) * np.cos(phi) * np.sin(phi) * 2
	#rhoB[N-1][0] = np.cos(theta) * np.sin(theta) * np.cos(phi) * np.sin(phi) * 2
    
    #W
    #start0itself = (np.sin(phi))**2 / (n-1)
    #start0another = np.sin(theta) * (np.sin(phi))**2 * ( ((n-2)/(n-1)**2) * np.sin(theta) + (2/(n-1)**(3/2)) * np.cos(theta) )
    #start1another = np.sin(theta) * np.sin(phi) * np.cos(phi) * (2*np.cos(theta) + (n-2)/np.sqrt(n-1) * np.sin(theta)) / (n-1)
    #start1itself = (np.cos(phi))**2
    #for v in range(n-1): #start0
     #   w = 2**v
      #  rhoB[w][w] = start0itself
       # for s in range(n-1): #start0
        #    u = 2**s
         #   if u != w:
          #      rhoB[w][u] = start0another
                #rhoB[u][w] = start0another
    #v = n-1 #start1
    #w = 2**v
    #rhoB[w][w] = start1itself
    #for s in range(n-1): #start0
     #   u = 2**s
      #  rhoB[w][u] = start1another
       # rhoB[u][w] = start1another
       
	#COMBO
	#cross = 1/2 * np.sin(theta) * np.cos(theta)
	#rhoB = np.full([N, N], cross)
	#r12 = 1/4
	#r03 = 1/4
	#rhoB[0][0] = r03
	#rhoB[1][1] = r12
	#rhoB[2][2] = r12
	#rhoB[3][3] = r03
	#rhoB[0][3] = r03
	#rhoB[3][0] = r03
	#rhoB[1][2] = r12
	#rhoB[2][1] = r12
      
	#rhoB = rhoA
    #print(np.nonzero(np.around((np.subtract(rhoA, rhoB)), 0)))
    
	Ps = np.diagonal(rhoA)
	print('START CHECK: total probability =')
	print(np.sum(Ps))
	Ps = np.diagonal(rhoB)
	print('START CHECK: total probability =')
	print(np.sum(Ps))
	Ps = None
    
	#AB    
	rho = np.kron(rhoA, rhoB)
	rhoA=None
	rhoB=None

	P = CSWAP(n, rho)
	return(P)

############################# INPUT: NUMBER OF QUBITS n
n = 4
N = 2**n

# create arrays
r = 1#20#32 # SET TO r=30 or 32 FOR FITS
#states(n, 0, 0)
theta = [0] * r
phi = [0] * r

P = [0] * r
Pent = [0] * r
Ps = [0] * r
P2 = [0] * r
P0 = [0] * r
Peven = [0] * r
Podd = [0] * r
P4 = [0] * r

# set arrays
popts = [0] * r
pcovs = [0] * r
#for j in range(r):
#	phi[j] = j*0.05 #j*0.03 + 0.34
c = ["purple", "red", "orange", "green", "blue"]
odd = [1,2,4,7,8,11,13,14,16,19,21,22,25,26,28,31]
phis = [0, np.pi/16, np.pi/8, np.pi*3/16, np.pi/4]
#phis = [0, np.pi/8, np.pi/4, np.pi*6/16, np.pi/2]
#phis=[0,0.2,0.4,0.6,0.8,1]
#phis=[0/100,0.2/100,0.4/100,0.6/100,0.8/100,1/100]
#phis=[0/10,0.2/10,0.4/10,0.6/10,0.8/10,1/10]
#phis=[np.pi/2]
phis=[0]

def f(data, x, y, z):
	#theta = data[0,:]
	#phi = data[1,:]
	#phi = thisphi
	#X = (n+2)/n
	#delta = (data - (np.pi/4))
	####### INPUT (IF FITTING 2): create fit 'f' to test
	f = x*np.sin(2*data)**2 + z
	return f

a = np.zeros(len(phis))
#thisphi = 0
for l, phi in enumerate(phis):
	N = 2**n
	P0 = [0] * r
	Peven = [0] * r
	Podd = [0] * r
	thisphi = phi
	for i in range(r):
		theta[i] = i * (np.pi/(2*r))#i*0.1 #+ np.arccos(1/np.sqrt(n))
		s = states(n, theta[i], phi)
		P0[i] = s[0]
		for o in odd:
			if o < N:
			    Podd[i] += s[o]
		Peven[i] = 1 - P0[i] - Podd[i]
		Ps[i] = s[3] #12 #3
		#P2[i] = s[7]
		#P4[i] = s[15]  # n=4 only
	delta = theta#np.subtract(theta, [np.pi/4]*r)
	#plt.plot(delta, P0, 'x', color=c[n-2])
	model = Podd
	print(P0[i], Podd[i], Peven[i])
	label = thisphi
	#plt.plot(delta, model, label=label)
    #plt.plot(delta, Peven, color=c[n-2])
#plt.plot(delta, Podd, color='black')
#plt.legend(title="$\epsilon/\pi$")
#plt.plot(np.arccos(1/np.sqrt(n)), 3/8, 'o')
	#plt.xlabel("$\delta$")
	#plt.ylabel("$P$")
#plt.savefig("mixedW3.pdf")

	#data = np.array([theta, phis])
	data = theta

	popt,pcov = curve_fit(f, data, model)  # IF FITTING, COMMENT BACK IN
	print("popt", popt)                      # IF FITTING, COMMENT BACK IN
	a[l] = popt[0]
	fit = np.zeros(r)
	for i in range(r):
		fit[i] = f(theta[i], popt[0], popt[1], popt[2])
	#label = thisphi#[thisphi/np.pi, popt]
	plt.plot(delta, fit, 'x')
	#plt.plot(np.arccos(1/np.sqrt(3)), 1/3, 'X')
	#y = [0] * len(delta)
	#for i, d in enumerate(delta):
	#    y[i] = (n-1)/(2*n) * ((np.cos(((n+2)/n) *d))**2)
	#plt.plot(delta, y, 'o')
	plt.legend()
plt.savefig("current.pdf")

#data = np.array([theta, phi])  # use if both A and B need to vary    
# GUESS EXPRESSION
def f2(data, x, y, z):
    #theta = data[0,:]
    #phi = data[1,:]
    
    ###### INPUT (IF FITTING): create fit 'f' to test
    #f = x * (np.cos(y * data))**a * (np.sin(b * data))**c + 0.25
    #p=data
    f = x*np.sin(2*data)**2 + z
    return f

if (len(phis) > 1):    
	try:
		popt2,pcov = curve_fit(f2, phis, a)
		print("a", popt2)
		pass
	except RuntimeError:
		print("Error - curve_fit failed")

	fit2 = np.zeros(len(phis))
	for i, j in enumerate(phis):
		fit2[i] = f2(j, popt2[0], popt2[1], popt2[2])
	plt.figure()
	plt.plot(phis, fit2, '.', label='two')
	plt.plot(phis, a)
	plt.savefig("current2.pdf")
