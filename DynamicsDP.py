from numpy import array, sin, cos

def D(q, params):
    alpha_1, alpha_2 = q
    
    l, m, J, b, g = params
    
    lc = l[0]/2, l[1]/2
    
    d11 = m[0]*lc[0]**2 + m[1]*(l[0]**2 + lc[1]**2 + 2*l[0]*lc[1]*cos(alpha_2)) + J[0] + J[1]
    d12 = m[1]*(lc[1]**2 + l[0]*lc[1]*cos(alpha_2)) + J[1]
    d21 = d12
    d22 = m[1] * lc[1]**2 + J[1]
    
    return array([[d11,d12],[d21,d22]])

def c_term(q, dq, params):
    alpha_1, alpha_2 = q
    dalpha_1, dalpha_2 = dq
    
    l, m, J, b, g = params
    
    lc = l[0]/2, l[1]/2

    hh = m[1]*l[0]*lc[1]*sin(alpha_2)
    
    c11 = -hh*dalpha_2*dalpha_1 - hh*(dalpha_1 + dalpha_2)*dalpha_2
    c21 = hh*dalpha_1**2
    
    return array([c11,c21])

def g_term(q, params):
    alpha_1, alpha_2 = q
    
    l, m, J, b, g = params
    
    lc = l[0]/2, l[1]/2
    
    g1 = (m[0]*lc[0]+m[1]*l[0])*g*sin(alpha_1) + m[1]*lc[1]*g*sin(alpha_1 + alpha_2)
    g2 = m[1]*g*lc[1]*sin(alpha_1 + alpha_2)
    
    return array([g1, g2])

def Q_d(q, dq, params):
    l, m, J, b, g = params

    dalpha_1, dalpha_2 = dq
    
    Q_d_1 = b[0]*dalpha_1
    Q_d_2 = b[1]*dalpha_2
    
    return array([Q_d_1, Q_d_2])

def h(q, dq, params):
    return c_term(q, dq, params) + g_term(q, params)