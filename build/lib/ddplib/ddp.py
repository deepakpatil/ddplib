#Author: Deepak Patil
#Toolbox is still under development.

import scipy.linalg as spl
import numpy as np
import control as ctrl

machine_epsilon = np.finfo(float).eps  

def ker(S):
    """S is a matrix to be entered as an array. This function returns the nullspace of matrix S. A key difference added is that null_space function in scipy returns an empty matrix on passing a full column rank matrix as an input. However, a non-singular matrix null-space is not empty. Its a singleton set with zero vector in it. This function returns a zero vector upon encountering a matrix which is a full column rank matrix."""
    V = spl.null_space(S)
    if V.shape[1] == 0:
       V = np.zeros((S.shape[1],1))

    return V
    
def subspace_sum(V1,V2):
    '''Calculates the sum of two subspaces im V1 and im V2.'''
    V1V2 = np.append(V1,-V2, axis=1)
    mv1 = V1.shape[1]
    Q1, R1, e1 = spl.qr(V1V2,pivoting = 'True')
    Q1 = np.matrix(Q1)
    ssum =np.empty((Q1.shape[0],0),float)
    for i in range(min(R1.shape)):
        if abs(R1[i,i])<100*machine_epsilon:
           pass
        else:
           ssum = np.append(ssum, Q1[:,i], axis =1)
    return ssum
    
def subspace_intersect(V1,V2):
    '''Calculates the intersection of two subspaces im V1 and im V2.'''
    V1V2 = np.append(V1,-V2, axis=1)
    mv1 = V1.shape[1]
    W = ker(V1V2)
    V1uV2 = V1.dot(W[:mv1,:])
    Q, R, e  = spl.qr(V1uV2,pivoting = True)
    Q = np.matrix(Q)
    #V1uV2p = V1uV2[:,e]
    ssi = np.empty((Q.shape[0],0),float)
    for i in range(min(R.shape)):
        if abs(R[i,i])<100*machine_epsilon:
           pass
        else:
           ssi = np.append(ssi, Q[:,i], axis =1)
           
    if ssi.shape[1] == 0:
       ssi = np.zeros((ssi.shape[0],1))

    return ssi

def subspace_sum_intersect(V1,V2):
    '''Calculates the sum and intersection of two subspaces im V1 and im V2.'''
    V1V2 = np.append(V1,-V2, axis=1)
    mv1 = V1.shape[1]
    W = ker(V1V2)
    V1uV2 = V1.dot(W[:mv1,:])
    Q, R, e  = spl.qr(V1uV2,pivoting = True)
    Q = np.matrix(Q)
    #V1uV2p = V1uV2[:,e]
    ssi = np.empty((Q.shape[0],0),float)
    for i in range(min(R.shape)):
        if abs(R[i,i])<100*machine_epsilon:
           pass
        else:
           ssi = np.append(ssi, Q[:,i], axis =1)
    Q1, R1, e1 = spl.qr(V1V2,pivoting = 'True')
    Q1 = np.matrix(Q1)
    ssum =np.empty((Q1.shape[0],0),float)
    for i in range(min(R1.shape)):
        if abs(R1[i,i])<100*machine_epsilon:
           pass
        else:
           ssum = np.append(ssum, Q1[:,i], axis =1)
    if ssi.shape[1] == 0:
       ssi = np.zeros((ssi.shape[0],1))

    return ssum,ssi

def basis_completion(V):
    '''Input: matrix of basis of subspace V, Output: completed basis '''
    Q, R, e = spl.qr(V,pivoting = True)
    if np.linalg.matrix_rank(V) < min(V.shape):
       v = np.empty((Q.shape[0],0),float)
       for i in range(min(R.shape)):
           if abs(R[i,i])<100*machine_epsilon:
              pass
           else:
              v = np.append(v, V[:,e[i]], axis =1)
       nv,mv = v.shape
    else:
       v = V[:,:R.shape[1]]
    
    ortho_comple = ker(v.transpose())
    full_basis = np.append(v, ortho_comple, axis=1)
    return full_basis

    
def matrix_eqn_solve(A,B):
    """AX=B all solutions X=X_ker*Z+X_part. Returns empty particular solution if not solvable. The current version is not an efficient piece of code. Needs rewriting for making it more efficient!"""
    A = np.matrix(A)
    B = np.matrix(B)
    n,p = A.shape
    m = B.shape[1]
    X_ker = ker(A)
    X_part = np.zeros((p,m))
    for i in range(m):
        Ab = np.append(A,B[:,i],axis=1)
        if np.linalg.matrix_rank(Ab)>np.linalg.matrix_rank(A):
           X_part = []
           break
        else:
           X = spl.lstsq(A,B[:,i])
           X_part[:,i] = np.reshape(X[0],(p,))

    return (X_ker, X_part)

def quotient(V1,V2):
    '''returns a subspace W such that V1 = V2 \directsum W'''
    T = matrix_eqn_solve(V1,V2)[1]
    Tbar = basis_completion(T)
    W = V1.dot(Tbar[:,V2.shape[1]:])
    return W

def affineintersect(S_ker1, S_part1, S_ker2, S_part2):
    ''' This function takes two affine spaces and computes the intersection affine space. Affine space is specified by matrices A, B are of appropriate size such that AX+B generates points in the affine space by arbitrary varying X.'''

    A=np.append(S_ker1, -S_ker2,axis=1) 
    B=S_part2-S_part1
    Sol_ker,Sol_part=matrix_eqn_solve(A,B)
    n=S_ker1.shape[1]
    p=S_part1.shape[1]
    X_ker=Sol_ker[0:n,:]
    X_part=Sol_part[0:n,:]
    Fcom_part=S_ker1.dot(X_part)+S_part1
    Fcom_ker=S_ker1.dot(X_ker)
    return Fcom_ker, Fcom_part;

def maximal_contrl_invariant_space(A,B,H):
    '''Computes maximal controlled invariant set for xdot = Ax + Bu, contained in ker H. '''
    V = ker(H)
    V_temp = np.empty((V.shape[0],0),float);
    rank_V_temp = 0
    while rank_V_temp != np.linalg.matrix_rank(V):
        V_temp = V
        rank_V_temp = np.linalg.matrix_rank(V)
        Z_1 = V_temp.transpose()
        Z_2 = B.transpose()
        Z = ker(np.append(Z_1,Z_2,axis = 0))
        H2 = Z.transpose().dot(A)
        V = ker(np.append(H,H2,axis = 0))
        
    return V

def matrix_equation(A,B,C,typ):
    '''Computes set of all solutions to BY + XA = C. The output is a list X_ker, Y_ker and particular solution X_part, Y_part. A general solution is then of the form 'linear combination of matrices in X_ker (resp. Y_ker) + X_part (resp. Y_part)'. Note that same linear combination should be used for both X_ker and Y_ker in generating one solution X and Y'''
    n = A.shape[0]
    m = A.shape[1]
    nb = B.shape[0]
    mb = B.shape[1]
    n1 = C.shape[0]
    n2 = m
    A1 = np.kron(A.transpose(), np.identity(n1))
    B1 = np.kron(np.identity(n2),B)
    AA = np.append(A1,B1,axis = 1)
    nn = n1*n2
    #vectorization operation is rows stacked upon one another
    size_of_vecX = n*n1
    BB = C.transpose().reshape([nn,1])
    x_ker, x_part = matrix_eqn_solve(AA,BB)
    vecX_ker = x_ker[:size_of_vecX,:]
    vecY_ker = x_ker[size_of_vecX:,:]
    vecX_part = x_part[:size_of_vecX,:]
    vecY_part = x_part[size_of_vecX:,:]
    #devectorization of kernel
    X_ker = []
    Y_ker = []
    for i in range(vecX_ker.shape[1]):
        if ~(np.all(x_ker[:,i]==0)):
           X_ker.append(vecX_ker[:,i].reshape([n,n1]).transpose())
           Y_ker.append(vecY_ker[:,i].reshape([m,mb]).transpose())
    if len(x_part) == 0: #no solution case
       X_part = [] 
       Y_part = []
    else:
       #devectorization of particular solution
       X_part = vecX_part.reshape([n,n1]).transpose()
       Y_part = vecY_part.reshape([m,mb]).transpose()

    return X_ker, X_part, Y_ker, Y_part


def set_of_friends(V,A,B):
    '''Computes a set of friends for a given (A,B)-invariant space V. It is returned in a parametric form F_ker X + F_part. F_ker is a list of matrices.'''

    RHS = A.dot(V)
    LHS = np.append(V,B,axis = 1)
    X_ker, X_part = matrix_eqn_solve(LHS,RHS)
    n = V.shape[0]
    m = V.shape[1]
    '''AV = [V B][X;U] => U = FV => FV+Uker Y = Upart => (V' kron I_rowsF, I_colY kron Uker)[vec F; vec Y] = vec Upart '''
    U_part = X_part[m:,:]
    U_ker = X_ker[m:,:]
    F_ker, F_part, Y_ker,Y_part= matrix_equation(V,U_ker,U_part,1)
    return F_ker, -F_part

def is_ddp_solvable(A,B,H,E):
    '''Given xdot = Ax + Bu + Ed, y = Hx, disturbance decoupling problem is solvable if and only if image of E is in V, the maximal controlled invariant subspace in ker H. This function checks if this condition is met and returns 1 if solvable and 0 if not met. '''
    V = maximal_contrl_invariant_space(A,B,H)
    VE = np.append(V,E,axis = 1)
    if np.linalg.matrix_rank(VE) == np.linalg.matrix_rank(V):
        return 1, V
    else:
        return 0, V

def kryl(A,B):
    nb,mb  = B.shape
    na,ma = A.shape
    if mb == 1:
       ctrbmat1 = B
       ctrbmat = ctrbmat1/np.linalg.norm(ctrbmat1,2)
       ctrbmat = ctrbmat.reshape((na,1))
       for i in range(na-1):
           ci = A.dot(ctrbmat[:,i])
           ci1 = ci/np.linalg.norm(ci,2)
           ci=ci1
           ctrbmat = np.append(ctrbmat,ci,axis=1)
    else:
       ctrbmat = np.empty((B.shape[0],0),float)
       for j in  range(mb):
           ctrbmat = np.append(ctrbmat,B[:,j]/np.linalg.norm(B[:,j],2),axis=1)

       for i in range(na-1):
           mc = ctrbmat.shape[1]
           bi1 = A.dot(ctrbmat[:,(mc-mb):])
           bi = np.empty((nb,0),float)
           for j in range(mb):
               bi = np.append(bi,bi1[:,j]/(np.linalg.norm(bi1[:,j],2)),axis=1)
           ctrbmat = np.append(ctrbmat,bi,axis=1)
    return ctrbmat
           


def controllability_subspace(A,B,H):
    V = maximal_contrl_invariant_space(A,B,H)
    S = np.zeros((A.shape[0],1))
    ASB = subspace_sum_intersect(A.dot(S),B)[0]
    S1 = subspace_sum_intersect(V,ASB)[1]
    SS1= np.append(S,S1,axis=1)
    while numpy.linalg.matrix_rank(S)<numpy.linalg.matrix_rank(SS1):
          S = S1
          ASB = subspace_sum_intersect(A.dot(S),B)[0]
          S1 = subspace_sum_intersect(V,ASB)[1]
          SS1= np.append(S,S1,axis=1)
 
    return S1    

def ddp_place(A,B,S,roots):
    '''Input: A,B matrices, S controllability subspace, roots of desired characteristic polynomial. Output: F such that eigenvalues of A+BF are roots and (A+BF)R\subset R'''
    nr, mr = S.shape 
    p = roots[:mr]
    q = roots[mr:]

    RsB = subspace_sum_intersect(S,B)[1]  # Rs \int B
    Z = quotient(B,RsB) #(Rs \int B) \ds Z = B
    RspZ = subspace_sum_intersect(S,Z)[0] #Rs \ds Z 
    W1 = quotient(np.identity(148),RspZ) #(Rs \ds Z) \ds W1 = Rn
    W = subspace_sum_intersect(Z,W1)[0] #W = Z \ds W1
    WQ = np.linalg.qr(W)[0]
    T2 = np.append(SQ,WQ,axis=1)  #T = Rs \ds W

    Anew = np.linalg.inv(T2).dot(A).dot(T2)
    Bnew = np.linalg.inv(T2).dot(B)

    nrsb,mrsb=RsB.shape
    A11 = Anew[:mr,:mr]
    A12 = Anew[:mr,mr:]
    A22 = Anew[mr:,mr:]
    A21 = Anew[mr:,:mr]
    B1 = Bnew[:mr,:mrsb]
    B2 = Bnew[mr:,mrsb:]

    F21 = spl.lstsq(B2,-A21)[0]
    F11 = ctrl.place(A11,B1,p)
    F22 = ctrl.place(A22,B2,q)
    F12 = np.zeros((F11.shape[0],F22.shape[1]))
    F1 = np.append(-F11,F12,axis=1)
    F2 = np.append(F21,-F22,axis=1)	
    F = np.append(F1,F2,axis=0)
    return F.dot(np.linalg.inv(T2))

 

'''Test Examples (from Linear Multivariable Control: A Geometric Approach by W Murray Wonham, Chapter 4, excercise problem 4.2, 4.8):
   #1
   A = np.matrix([[0, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0]])
   B = np.matrix([[0,0],[0,0],[1,0],[0,0],[0,1]])
   H = np.matrix([[1,0,0,-1,0],[1,-1,0,0,0],[0,0,0,1,-1]])

   V = span of [1;1;1;1;1] (column). Output will be normalized version!
   Check if (A+B.dot(F_part+F_ker[i]).dot(V) is multiple of V

   #2 
   A = np.matrix([[0, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0]])
   B = np.matrix([[0,0],[0,0],[1,0],[0,1],[0,0]])
   H = np.matrix([[1,0,0,0,0],[0,0,0,1,0]])

   V = span of [0;0;0;0;1] (column)
   Check if (A+B.dot(F_part+F_ker[i]).dot(V) is multiple of V '''
