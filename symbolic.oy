import sympy as sp

kx = sp.symbols('kx')
x = 0.

M = sp.Matrix([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
M[0, 0] = 1. 
M[0, 1] = 2./3.
M[0, 2] = 2./3.
M[1, 0] = sp.exp(1j*kx) * 1./6. + x
M[1, 1] = sp.exp(1j*kx) * 2./3.
M[1, 2] = sp.exp(1j*kx) * -1./3.
M[2, 0] = sp.exp(-1j*kx) * 1./6.
M[2, 1] = sp.exp(-1j*kx) * -1./3.
M[2, 2] = sp.exp(-1j*kx) * 2./3.

dict_eig = M.eigenvals()
