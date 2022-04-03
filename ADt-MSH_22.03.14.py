# -*- coding: utf-8 -*-
"""
@autor: jp
23.02.22

Organização do programa.
Nesta versão será revisado todo o código utilizando apenas o atenuador pendular. As equações de equilíbrio
dinâmico acopladas formam o seguinte sistema:

M ⋅ ä(t) + mₚ ⋅ ä(t) - mₚ ⋅ Lₚ ⋅ sen[o(t) ⋅ ȯ²(t)] + mₚ ⋅ Lₚ ⋅ cos[o(t) ⋅ ȯ(t)] + K ⋅ a(t) + α ⋅ M ⋅ ȧ(t) = q(t)
mₚ ⋅ ä(t) ⋅ Lₚ ⋅ cos[o(t)] + mₚ ⋅ Lₚ² ⋅ ö(t) + mₚ ⋅ g ⋅ Lₚ ⋅ sen[o(t)] + cₚ ⋅ ȯ(t) + kₚ ⋅ o(t) = 0

Onde:

α - coef. de proporcionalidade massa-amortecimento
cₚ - amortecimento do pêndulo: ξₚ = x %
g - aceleração da gravidade: g = 9,80665 m/s²
K - rigidez modal da estrutura: K = M ⋅ ω²
Lₚ - comprimento do pêndulo: L = g /(4 ⋅ π² ⋅ fₚ²)
M - massa modal da estrutura: M = 2610,996 kg
mₚ - massa do pêndulo: mₚ = x kg
a(t) - coordenada modal
ȧ(t) - primeira derivada da coord. modal
ä(t) - segunda derivada da coord. modal
o(t) - deslocamento angular do pêndulo
ȯ(t) - velocidade angular do pêndulo
ö(t) - aceleração angular do pêndulo
q(t) - força modal do vento
ψ - razão entre a frequência do atenuador e da estrutura: ψ = ω/ωₙ = 1/(1 + μ)
μ - razão entre a massa do atenuador e da estrutura: μ = m/mₙ

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import re
from scipy import interpolate
import time
start_time = time.time()
from datetime import datetime
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")

def vetorDeTempo(ti, tf, dt):                   # Definição do vetor de amostras de tempo
    N = int(1 + (tf - ti) / dt)
    t = np.zeros(N)
    for i in range(1, N):
        t[i] = t[i-1] + dt
    return t

def u_n(t, freqInfo, estVenInfo):               # Cálculo das flutuações uᵢ(t) para i = 1, 2, ..., n discretizações da estrutura
    '''
    print('\nInício do cálculo de uᵢ(t), i = 1, 2, ..., n')
    now = datetime.now()
    print(now.strftime("%d/%m/%Y %H:%M:%S"))
    '''
    f = freqInfo[0]
    f_i = freqInfo[1]
    f_f = freqInfo[2]
    m = freqInfo[3]
    n = estVenInfo[6]
    delta_f = (freqInfo[2] - freqInfo[1])/m
    u_n = []
    for i in range(n):
        # Definição do espectro de Davenport reduzido Sr(f)
        f = freqInfo[0]
        U_0 = 24.11                             # Velocidade característica na altura z = 10 m
        z_0 = 2.5                               # Comprimento de rugosidade do terreno (NBR 6123)
        u_fric = 0.4*U_0/(np.log(10/z_0))       # Velocidade de fricção
        def S_r(f): return 4*(((1200*f/U_0)**2)/((1+(1200*f/U_0)**2)**(4/3)))

        # Limites de integração iₖ em escala logarítmica, sendo i = 1, 2, ..., m+1
        i_k = np.geomspace(freqInfo[1], freqInfo[2], m+1)
        
        # Definição das frequências fₖ em escala logarítmica
        f_k = np.zeros(m)
        for j in range(1, m+1):
            exec(f'exec("f%d = %s" % (j, np.sqrt(i_k[{j-1}]*i_k[{j}])))')
            exec(f'f_k[{j-1}] = f{j}')

        # Cálculo de Cₖ
        for j in range(0, m):
            exec(f'exec("C%d = %s" % (j+1, u_fric*np.sqrt(2*S_r(f{j+1})*delta_f)))')

        # Definição dos ângulos de fase aleatórios θₖ
        def teta_k(): return np.random.uniform(0,2*np.pi)
        for j in range(0, m):
            exec("teta%d = %s" % (j+1, teta_k()))

        # Cálculo de uₖ(t)
        for j in range(0, m):
            exec(f"u{j+1} = C{j+1}*np.cos(np.add(2*float(np.pi)*(f{j+1})*t, teta{j+1}))")

        # Cálculo de u(t)
        uu0 = np.zeros(len(t))
        for j in range(0, m):
            exec(f"uu{j+1} = np.add(uu{j}, u{j+1})")

        exec(f'u_{i+1} = eval("uu%d" % (m))')

        '''
        exec(f'print("\\nu_{i+1}(t) =", u_{i+1})')
        exec(f'print("\\nCálculo de u_{i+1}(t) ✓")')
        now = datetime.now()
        print(now.strftime("%d/%m/%Y %H:%M:%S"))
        '''
        exec(f'u_n.append(u_{i+1})')
    return u_n

def q(t, estVenInfo, u_n):                      # Cálculo da força modal q(t)
    n = estVenInfo[6]
    pondELS = estVenInfo[0]
    for i in range(0, n):
        exec(f"q{i+1} = 0.5*estVenInfo[1][{i}]*estVenInfo[2][{i}]*(1*estVenInfo[3][{i}]**2 + 2*estVenInfo[3][{i}]*u_n[{i}])")
        # qᵢ(t) = Cₐᵢ ⋅ Aᵢ ⋅ ɸᵢ ⋅ (Ūᵢ² + 2 ⋅ Ūᵢ ⋅ uᵢ(t))/2

    qq0 = np.zeros(len(t))
    for i in range(0, n):
        exec(f"qq{i+1} = np.add(qq{i}, q{i+1})")

    q = eval("qq%d" % (n))*estVenInfo[4]  # q(t) = ρ ⋅ Σⁿᵢ₌₁ qᵢ(t)
    '''
    estVenInfo = [pondELS, Ca_ixAe_i, phi_i, U_i, ro, M, n, xi, omega]
    '''
    
    t_ = int(1 + (60 - ti) / dt)        # Suavização dos segundos iniciais para evitar efeito de impacto
    i = 0
    while i <= t_:
        q[i] = np.tanh(15*i/t_)*q[i]
        i += 1

    return pondELS*q

def alpha(estVenInfo):                          # Cálculo de α = 2 ⋅ ξ ⋅ ω + ρ ⋅ Σⁿᵢ₌₁ Cₐᵢ ⋅ Aᵢ ⋅ ɸ²ᵢ ⋅ Ūᵢ/m
    n = estVenInfo[6]
    for i in range(0, n):
        exec(f"aa{i+1} = estVenInfo[1][{i}]*estVenInfo[2][{i}]**2*estVenInfo[3][{i}]")
        # α'ᵢ = Cₐᵢ ⋅ Aᵢ ⋅ ɸ²ᵢ ⋅ Ūᵢ
        # estVenInfo = [pondELS, Ca_ixAe_i, phi_i, U_i, ro, M, n, xi, omega]

    aaa0 = 0
    for i in range(0, n):
        exec(f"aaa{i+1} = np.add(aaa{i}, aa{i+1})")

    alpha_aero = eval("aaa%d * estVenInfo[4]/estVenInfo[5]" % (n))  # αₐₑᵣ = ρ ⋅ Σⁿᵢ₌₁ α'ᵢ/M
    alpha_estr = 2*estVenInfo[7]*estVenInfo[8]                      # αₑₛₜ = 2 ⋅ ξ ⋅ ω
    alpha = alpha_aero + alpha_estr     # α = αₑₛₜ + αₐₑᵣ
    
    #print('αₐₑᵣₒ =', round(alpha_aero,2))
    #print('αₑₛₜᵣ =', round(alpha_estr, 2))
    #print('α =', round(alpha, 2))

    return alpha

def eqDif1(y, q, alpha, estVenInfo):            # Equação do equilíbrio dinâmico para a estrutura livre (não controlada)
    '''
    ä(t) = -α ⋅ ȧ(t) - ω² ⋅ a(t) + q(t)/m
    '''

    # Condições de contorno: [d, v]
    d, v = y
    
    Dy = np.array([v, -1*alpha*v -1*estVenInfo[8]**2*d + q/estVenInfo[5]])
    '''
    estVenInfo = [pondELS, Ca_ixAe_i, phi_i, U_i, ro, M, n, xi, omega]
    '''

    return Dy

def eqDif2(y, q, alpha, estVenInfo, aten1Info): # Sistema de equações de equilíbrio dinâmico para a estrutura controlada pelo atenuador pendular
    '''
    Estrutura controlada por um atenuador pendular ajustado a ψ ⋅ fₙ

    M ⋅ ä(t) + mₚ ⋅ ä(t) - mₚ ⋅ Lₚ ⋅ sen[o(t) ⋅ ȯ²(t)] + mₚ ⋅ Lₚ ⋅ cos[o(t) ⋅ ȯ(t)] + K ⋅ a(t) + α ⋅ M ⋅ ȧ(t) = q(t)
    mₚ ⋅ ä(t) ⋅ Lₚ ⋅ cos[o(t)] + mₚ ⋅ Lₚ² ⋅ ö(t) + mₚ ⋅ g ⋅ Lₚ ⋅ sen[o(t)] + cₚ ⋅ ȯ(t) = 0
    '''
    m = aten1Info[0]
    L = aten1Info[1]
    c = aten1Info[2]
    g = aten1Info[3]
    k = aten1Info[4]
    M = estVenInfo[5]
    omega = estVenInfo[8]

    # Condições de contorno: [d_torre, d_pendulo, v_torre, v_pendulo]
    d, o, v, w = y
    
    Dy = np.array([v, w,
    ((1/(m*L*np.cos(o))) * ((-1*m*L**2*np.cos(o))/(m*L*(np.cos(o)**2-1)-M*L) * (g*np.tan(o)*(M+m) +
    k*o*(M+m)/(m*L*np.cos(o)) + c*w*(M+m)/(m*L*np.cos(o)) + m*L*w**2*np.sin(o)- omega**2*M*d - alpha*M*v + q) -m*g*L*np.sin(o) - k*o - c*w)),
    ((np.cos(o)/(m*L*(np.cos(o)**2-1)-M*L)) * (g*np.tan(o)*(M+m) + k*o*(M+m)/(m*L*np.cos(o)) + c*w*(M+m)/(m*L*np.cos(o)) + m*L*w**2*np.sin(o) -
    omega**2*M*d - alpha*M*v + q))])
    '''
    estVenInfo = [pondELS, Ca_ixAe_i, phi_i, U_i, ro, M, n, xi, omega]
    aten1Info = [m_p, L_p, c_p, g, k_p]
    '''

    return Dy

def eqDif3(y, q, alpha, estVenInfo, aten2Info): # Sistema de equações de equilíbrio dinâmico para a estrutura controlada pelo atenuador discoide
    '''
    Estrutura controlada por um atenuador discoide ajustado a ψ ⋅ 0,02 Hz

    M ⋅ ä(t) +  K ⋅ a(t) + kₐ ⋅ [a(t) - ė(t)] + α ⋅ M ⋅ ȧ(t) + cₐ ⋅ [ȧ(t) - ė(t)] = q(t) ⋅ M
    mₐ ⋅ ë(t) + cₐ ⋅ [ė(t) - ȧ(t)] + kₐ ⋅ [e(t) - a(t)] = 0
    '''
    m = aten2Info[0]
    k = aten2Info[1]
    c = aten2Info[2]
    M = estVenInfo[5]
    omega = estVenInfo[8]

    # Condições de contorno: [d_torre, d_pendulo, v_torre, v_pendulo]
    d, o, v, w = y
    
    Dy = np.array([v, w,
    ((q - (k*(d-o) + c*(v-w) + omega**2*M*d + alpha*M*v))/M),
    ((c*(v-w) + k*(d-o))/m)])
    '''
    estVenInfo = [pondELS, Ca_ixAe_i, phi_i, U_i, ro, M, n, xi, omega]
    aten2Info = [m_d, k_d, c_d]
    '''

    return Dy

def rk4_1(y0, t, q, alpha, estVenInfo):            # Runge-Kutta de 4ª ordem para estrutura livre (não controlada)
    
    N = len(t)                      # Número de amostras de tempo
    y = np.zeros([2, N])            # Matriz 2 x n a receber a solução
    y[:,0] = y0                     # Definição das condições de contorno em t = 0 na matriz solução
    h = t[1] - t[0]                 # Intervalo de tempo dt
    
    for i in range(0, N-1):

        k1 = h*eqDif1(y[:,i], q[i], alpha, estVenInfo)          # k₁ = h ⋅ f(yᵢ, tᵢ)
        k2 = h*eqDif1(y[:,i]+k1/2, q[i], alpha, estVenInfo)     # k₂ = h ⋅ f(yᵢ+k₁/2, tᵢ+h/2)
        k3 = h*eqDif1(y[:,i]+k2/2, q[i], alpha, estVenInfo)     # k₃ = h ⋅ f(yᵢ+k₂/2, tᵢ+h/2)
        k4 = h*eqDif1(y[:,i]+k3, q[i], alpha, estVenInfo)       # k₄ = h ⋅ f(yᵢ+k₃, tᵢ+h)

        y[:,i+1] = y[:,i]+(k1+2*k2+2*k3+k4)/6                   # yᵢ₊₁ = yᵢ + (k₁ + 2k₂ + 2k₃ + k₄)/6
    
    return y                        # Matriz solução: [d, v]

def rk4_2(y0, t, q, alpha, estVenInfo, aten1Info): # Runge-Kutta de 4ª ordem para estrutura controlada com atenuador pendular
    
    N = len(t)                      # Número de amostras de tempo
    y = np.zeros([4, N])            # Matriz 2 x n a receber a solução da torre
    y[:,0] = y0                     # Condições de contorno em t = 0 na matriz solução
    h = t[1] - t[0]                 # Intervalo de tempo dt

    for i in range(0, N-1):

        k1 = h*eqDif2(y[:,i], q[i], alpha, estVenInfo, aten1Info)        # k₁ = h ⋅ f(yᵢ, tᵢ)
        k2 = h*eqDif2(y[:,i]+k1/2, q[i], alpha, estVenInfo, aten1Info)   # k₂ = h ⋅ f(yᵢ+k₁/2, tᵢ+h/2)
        k3 = h*eqDif2(y[:,i]+k2/2, q[i], alpha, estVenInfo, aten1Info)   # k₃ = h ⋅ f(yᵢ+k₂/2, tᵢ+h/2)  
        k4 = h*eqDif2(y[:,i]+k3, q[i], alpha, estVenInfo, aten1Info)     # k₄ = h ⋅ f(yᵢ+k₃, tᵢ+h)

        y[:,i+1] = y[:,i]+(k1+2*k2+2*k3+k4)/6                           # yᵢ₊₁ = yᵢ + (k₁ + 2k₂ + 2k₃ + k₄)/6

    return y

def rk4_3(y0, t, q, alpha, estVenInfo, aten2Info): # Runge-Kutta de 4ª ordem para estrutura controlada com atenuador discoidal
    
    N = len(t)                      # Número de amostras de tempo
    y = np.zeros([4, N])            # Matriz 2 x n a receber a solução da torre
    y[:,0] = y0                     # Condições de contorno em t = 0 na matriz solução
    h = t[1] - t[0]                 # Intervalo de tempo dt

    for i in range(0, N-1):

        k1 = h*eqDif3(y[:,i], q[i], alpha, estVenInfo, aten2Info)        # k₁ = h ⋅ f(yᵢ, tᵢ)
        k2 = h*eqDif3(y[:,i]+k1/2, q[i], alpha, estVenInfo, aten2Info)   # k₂ = h ⋅ f(yᵢ+k₁/2, tᵢ+h/2)
        k3 = h*eqDif3(y[:,i]+k2/2, q[i], alpha, estVenInfo, aten2Info)   # k₃ = h ⋅ f(yᵢ+k₂/2, tᵢ+h/2)  
        k4 = h*eqDif3(y[:,i]+k3, q[i], alpha, estVenInfo, aten2Info)     # k₄ = h ⋅ f(yᵢ+k₃, tᵢ+h)

        y[:,i+1] = y[:,i]+(k1+2*k2+2*k3+k4)/6                           # yᵢ₊₁ = yᵢ + (k₁ + 2k₂ + 2k₃ + k₄)/6

    return y

def deslocEstat(a, dt):                             # Cálculo do deslocamento estático (médio) após o fim da suavização da força modal (45 s)
    a_med = np.zeros(int(round((600 - 10)/dt, 0)))
    j = 0
    for i in range(len(a)):
        if i > 10/dt:
            a_med[j] = a[i]
            j = j + 1
    
    return np.mean(a_med)

def salvarFig_nc(vetores):                      # Plotagem das soluções da EDO, da flutuação u(t) e da força modal q(t)
    t = vetores[0]
    a = vetores[1]
    a_ = vetores[2]
    a__ = vetores[3]
    u = vetores[4]
    q = vetores[5]

    plt.figure(1)                       # a(t)
    plt.style.use('classic')
    plt.grid(True)
    plt.ylabel("$x\,(t)\;\;[m]$\n", fontsize=16)
    plt.xlabel("\n$t\;\;[s]$", fontsize=16)
    plt.ylim([0, .2])
    plt.tick_params(axis='x', which='major', bottom=True, top=False, labelsize=13)
    plt.tick_params(axis='y', which='major', left=True, right=False, labelsize=13)
    plt.xticks([0, 150, 300, 450, 600], ['$0$', '$150$', '$300$', '$450$', '$600$'])
    plt.yticks([0, .05, .1, .15, .2], ['$0$', '$0,05$', '$0,10$', '$0,15$', '$0,20$'])
    exec(f'plt.text(400, .15, "$x_{{max}} = {round(max(a), 2)}\,m$\\n$t={int(round(t[np.argmax(a)], 0))}\,s$", fontsize=13, bbox=dict(facecolor="white", alpha=1))')
    plt.plot(t, a)
    plt.plot(t[np.argmax(np.abs(a))], max(a), 'wo')
    plt.tight_layout()
    exec(f"plt.savefig('./Plotagens/a(t), 0 a 600 s, estrutura livre.png', dpi=600)")
    #plt.show()
    plt.close()
    print("\na(t).png ✓")

    plt.figure(2)                       # ȧ(t)
    plt.style.use('classic')
    plt.grid(True)
    plt.ylabel("$\dot{{x}}\,(t)\;\;[m/s]$\n", fontsize=16)
    plt.xlabel("\n$t\;\;[s]$", fontsize=16)
    plt.ylim([-.3, .3])
    plt.tick_params(axis='x', which='major', bottom=True, top=False, labelsize=14)
    plt.tick_params(axis='y', which='major', left=True, right=False, labelsize=14)
    plt.xticks([0, 150, 300, 450, 600], ['$0$', '$150$', '$300$', '$450$', '$600$'])
    plt.yticks([-.3, -.15, 0, .15, .3], ['$-0,30$', '$-0,15$', '$0$', '$0,15$', '$0,30$'])
    exec(f'plt.text(400, 0.225, "$\dot{{x}}_{{max}} = {round(max(a_, key=abs), 2)}\,m/s$\\n$t={int(round(t[np.argmax(a_)], 0))}\,s$", fontsize=14, bbox=dict(facecolor="white", alpha=1))')
    plt.plot(t, a_)
    plt.plot(t[np.argmax(np.abs(a_))], max(a_), 'wo')
    plt.tight_layout()
    exec(f"plt.savefig('./Plotagens/ȧ(t), 0 a 600 s, estrutura livre.png', dpi=600)")
    #plt.show()
    plt.close()
    print("ȧ(t).png ✓")

    plt.figure(3)                       # ä(t)
    plt.style.use('classic')
    plt.grid(True)
    plt.ylabel("$\ddot{{x}}\,(t)\;\;[m/s^2]$\n", fontsize=16)
    plt.xlabel("\n$t\;\;[s]$", fontsize=16)
    plt.ylim([-1.5, 1.5])
    plt.tick_params(axis='x', which='major', bottom=True, top=False, labelsize=14)
    plt.tick_params(axis='y', which='major', left=True, right=False, labelsize=14)
    plt.xticks([0, 150, 300, 450, 600], ['$0$', '$150$', '$300$', '$450$', '$600$'])
    plt.yticks([-1.5, -1, -0.5, 0, 0.5, 1.0, 1.5], ['$-1,5$', '$-1,0$', '$-0,5$', '$0$', '$0,5$', '$1,0$', '$1,5$'])
    exec(f'plt.text(400, 1.125, "$\ddot{{x}}_{{max}} = {round(max(a__, key=abs), 2)}\,m/s^2$\\n$t={int(round(t[np.argmax(a__)], 0))}\,s$", fontsize=14, bbox=dict(facecolor="white", alpha=1))')
    plt.plot(t, a__)
    plt.plot(t[np.argmax(np.abs(a__))], max(a__), 'wo')
    plt.tight_layout()
    exec(f"plt.savefig('./Plotagens/ä(t), 0 a 600 s, estrutura livre.png', dpi=600)")
    #plt.show()
    plt.close()
    print("ä(t).png ✓")

    if np.sum(u) != 0:

        plt.figure(4)                       # u(t)
        plt.style.use('classic')
        plt.grid(True)
        plt.ylabel("$u\,(t)\;\;[m/s]$\n", fontsize=16)
        plt.xlabel("\n$t\;\;[s]$", fontsize=16)
        plt.ylim([-22, 22])
        plt.tick_params(axis='x', which='major', bottom=True, top=False, labelsize=14)
        plt.tick_params(axis='y', which='major', left=True, right=False, labelsize=14)
        plt.xticks([0, 150, 300, 450, 600], ['$0$', '$150$', '$300$', '$450$', '$600$'])
        plt.yticks([-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25], ['$-25$', '$-20$', '$-15$', '$-10$', '$-5$', '$0$', '$5$', '$10$', '$15$', '$-20$', '$-25$'])
        exec(f'plt.text(400, 16.5, "$u_{{max}} = {round(max(u, key=abs), 2)}\,m/s$\\n$t={int(round(t[np.argmax(u)], 0))}\,s$", fontsize=15, bbox=dict(facecolor="white", alpha=1))')
        plt.plot(t, u)
        plt.plot(t[np.argmax(np.abs(u))], max(u), 'wo')
        plt.tight_layout()
        exec(f"plt.savefig('./Plotagens/u(t), z = z₂₀, 0 a 600 s.png', dpi=600)")
        #plt.show()
        plt.close()
        print("u(t).png ✓")

    if np.sum(q) != 0:

        plt.figure(5)                   # q(t)
        plt.style.use('classic')
        plt.grid(True)
        plt.ylabel("$q\,(t)\;\;[N]$\n", fontsize=16)
        plt.xlabel("\n$t\;\;[s]$", fontsize=16)
        plt.ylim([0, 20])
        plt.tick_params(axis='x', which='major', bottom=True, top=False, labelsize=14)
        plt.tick_params(axis='y', which='major', left=True, right=False, labelsize=14)
        plt.xticks([0, 150, 300, 450, 600], ['$0$', '$150$', '$300$', '$450$', '$600$'])
        plt.yticks([0, 5, 10, 15, 20], ['$0$', '$5$', '$10$', '$15$', '$-20$'])
        exec(f'plt.text(400, 16.5, "$q_{{max}} = {round(max(q, key=abs)/1000, 2)}\,N$\\n$t={int(round(t[np.argmax(q)], 0))}\,s$", fontsize=14, bbox=dict(facecolor="white", alpha=1))')
        plt.plot(t, q/1000)
        plt.plot(t[np.argmax(np.abs(q))], max(q)/1000, 'wo')
        plt.tight_layout()
        exec(f"plt.savefig('./Plotagens/q(t), 0 a 600 s.png', dpi=600)")
        #plt.show()
        plt.close()
        print("q(t).png ✓\n")

def salvarFig_c(vetores):                       # Plotagem das soluções da EDO, da flutuação u(t) e da força modal q(t)
    t = vetores[0]
    a = vetores[1]
    a_ = vetores[2]
    a__ = vetores[3]
    o = vetores[4]
    o_ = vetores[5]
    o__ = vetores[6]
    u = vetores[7]
    q = vetores[8]

    plt.figure(1)                       # a(t)
    plt.style.use('classic')
    plt.grid(True)
    plt.ylabel("$x\,(t)\;\;[m]$\n", fontsize=16)
    plt.xlabel("\n$t\;\;[s]$", fontsize=16)
    plt.ylim([0, .2])
    plt.tick_params(axis='x', which='major', bottom=True, top=False, labelsize=13)
    plt.tick_params(axis='y', which='major', left=True, right=False, labelsize=13)
    plt.xticks([0, 150, 300, 450, 600], ['$0$', '$150$', '$300$', '$450$', '$600$'])
    plt.yticks([0, .05, .1, .15, .2], ['$0$', '$0,05$', '$0,10$', '$0,15$', '$0,20$'])
    exec(f'plt.text(400, .15, "$x_{{max}} = {round(max(a), 2)}\,m$\\n$t={int(round(t[np.argmax(a)], 0))}\,s$", fontsize=13, bbox=dict(facecolor="white", alpha=1))')
    plt.plot(t, a)
    plt.plot(t[np.argmax(np.abs(a))], max(a), 'wo')
    plt.tight_layout()
    exec(f"plt.savefig('./Plotagens/a(t), 0 a 600 s, estrutura controlada.png', dpi=600)")
    #plt.show()
    plt.close()
    print("\na(t).png ✓")

    plt.figure(2)                       # ȧ(t)
    plt.style.use('classic')
    plt.grid(True)
    plt.ylabel("$\dot{{x}}\,(t)\;\;[m/s]$\n", fontsize=16)
    plt.xlabel("\n$t\;\;[s]$", fontsize=16)
    plt.ylim([-.3, .3])
    plt.tick_params(axis='x', which='major', bottom=True, top=False, labelsize=14)
    plt.tick_params(axis='y', which='major', left=True, right=False, labelsize=14)
    plt.xticks([0, 150, 300, 450, 600], ['$0$', '$150$', '$300$', '$450$', '$600$'])
    plt.yticks([-.3, -.15, 0, .15, .3], ['$-0,30$', '$-0,15$', '$0$', '$0,15$', '$0,30$'])
    exec(f'plt.text(400, 0.225, "$\dot{{x}}_{{max}} = {round(max(a_, key=abs), 2)}\,m/s$\\n$t={int(round(t[np.argmax(a_)], 0))}\,s$", fontsize=14, bbox=dict(facecolor="white", alpha=1))')
    plt.plot(t, a_)
    plt.plot(t[np.argmax(np.abs(a_))], max(a_, key=abs), 'wo')
    plt.tight_layout()
    exec(f"plt.savefig('./Plotagens/ȧ(t), 0 a 600 s, estrutura controlada.png', dpi=600)")
    #plt.show()
    plt.close()
    print("ȧ(t).png ✓")

    plt.figure(3)                       # ä(t)
    plt.style.use('classic')
    plt.grid(True)
    plt.ylabel("$\ddot{{x}}\,(t)\;\;[m/s^2]$\n", fontsize=16)
    plt.xlabel("\n$t\;\;[s]$", fontsize=16)
    plt.ylim([-6.25, 6.25])
    plt.tick_params(axis='x', which='major', bottom=True, top=False, labelsize=14)
    plt.tick_params(axis='y', which='major', left=True, right=False, labelsize=14)
    plt.xticks([0, 150, 300, 450, 600], ['$0$', '$150$', '$300$', '$450$', '$600$'])
    plt.yticks([-1.5, -1, -0.5, 0, 0.5, 1.0, 1.5], ['$-1,5$', '$-1,0$', '$-0,5$', '$0$', '$0,5$', '$1,0$', '$1,5$'])
    exec(f'plt.text(400, 1.125, "$\ddot{{x}}_{{max}} = {round(max(a__, key=abs), 2)}\,m/s^2$\\n$t={int(round(t[np.argmax(a__)], 0))}\,s$", fontsize=14, bbox=dict(facecolor="white", alpha=1))')
    plt.plot(t, a__)
    plt.plot(t[np.argmax(np.abs(a__))], max(a__, key=abs), 'wo')
    plt.tight_layout()
    exec(f"plt.savefig('./Plotagens/ä(t), 0 a 600 s, estrutura controlada.png', dpi=600)")
    #plt.show()
    plt.close()
    print("ä(t).png ✓")

    plt.figure(4)                       # θ(t)
    plt.style.use('classic')
    plt.grid(True)
    plt.ylabel("$\\theta\,(t)\;\;[rad]$\n", fontsize=16)
    plt.xlabel("\n$t\;\;[s]$", fontsize=16)
    plt.ylim([-0.3, .3])
    plt.tick_params(axis='x', which='major', bottom=True, top=False, labelsize=12)
    plt.tick_params(axis='y', which='major', left=True, right=False, labelsize=12)
    plt.xticks([0, 150, 300, 450, 600], ['$0$', '$150$', '$300$', '$450$', '$600$'])
    plt.yticks([-.3, -.15, 0, .15, .3], ['$-0,30$', '$-0,15$', '$0$', '$0,15$', '$0,30$'])
    exec(f'plt.text(400, .15, "$\\\\theta_{{max}} = {round(max(o), 2)}\,rad$\\n$t={int(round(t[np.argmax(o)], 0))}\,s$", fontsize=13, bbox=dict(facecolor="white", alpha=1))')
    plt.plot(t, o)
    plt.plot(t[np.argmax(np.abs(o))], max(o, key=abs), 'wo')
    plt.tight_layout()
    exec(f"plt.savefig('./Plotagens/θ(t), 0 a 600 s, sistema de controle.png', dpi=600)")
    #plt.show()
    plt.close()
    print("\nθ(t).png ✓")

    plt.figure(5)                       # ω(t)
    plt.style.use('classic')
    plt.grid(True)
    plt.ylabel("$\omega\,(t)\;\;[rad/s]$\n", fontsize=16)
    plt.xlabel("\n$t\;\;[s]$", fontsize=16)
    plt.ylim([-1.35, 1.35])
    plt.tick_params(axis='x', which='major', bottom=True, top=False, labelsize=14)
    plt.tick_params(axis='y', which='major', left=True, right=False, labelsize=14)
    plt.xticks([0, 150, 300, 450, 600], ['$0$', '$150$', '$300$', '$450$', '$600$'])
    plt.yticks([-1.2, -.8, -.4, 0, .4, .8, 1.2], ['$-1,2$', '$-0,8$', '$-0,40$', '$0$', '$0,40$', '$0,8$', '$1,2$'])
    exec(f'plt.text(400, 1.0125, "$\omega_{{max}} = {round(max(o_, key=abs), 2)}\,rad/s$\\n$t={int(round(t[np.argmax(o_)], 0))}\,s$", fontsize=14, bbox=dict(facecolor="white", alpha=1))')
    plt.plot(t, o_)
    plt.plot(t[np.argmax(np.abs(o_))], max(o_, key=abs), 'wo')
    plt.tight_layout()
    exec(f"plt.savefig('./Plotagens/ω(t), 0 a 600 s, sistema de controle.png', dpi=600)")
    #plt.show()
    plt.close()
    print("ω(t).png ✓")

    plt.figure(6)                       # α(t)
    plt.style.use('classic')
    plt.grid(True)
    plt.ylabel("$\\alpha\,(t)\;\;[rad/s^2]$\n", fontsize=16)
    plt.xlabel("\n$t\;\;[s]$", fontsize=16)
    plt.ylim([-6.25, 6.25])
    plt.tick_params(axis='x', which='major', bottom=True, top=False, labelsize=14)
    plt.tick_params(axis='y', which='major', left=True, right=False, labelsize=14)
    plt.xticks([0, 150, 300, 450, 600], ['$0$', '$150$', '$300$', '$450$', '$600$'])
    plt.yticks([-6, -4, -2, 0, 2, 4, 6], ['$-6,0$', '$-4,0$', '$-2,0$', '$0$', '$2,0$', '$4,0$', '$6,0$'])
    exec(f'plt.text(400, 4.6875, "$\\\\alpha_{{max}} = {round(max(o__, key=abs), 2)}\,rad/s^2$\\n$t={int(round(t[np.argmax(o__)], 0))}\,s$", fontsize=14, bbox=dict(facecolor="white", alpha=1))')
    plt.plot(t, o__)
    plt.plot(t[np.argmax(np.abs(o__))], max(o__, key=abs), 'wo')
    plt.tight_layout()
    exec(f"plt.savefig('./Plotagens/α(t), 0 a 600 s, sistema de controle.png', dpi=600)")
    #plt.show()
    plt.close()
    print("α(t).png ✓")

    if np.sum(u) != 0:

        plt.figure(7)                       # u(t)
        plt.style.use('classic')
        plt.grid(True)
        plt.ylabel("$40\\%\cdot u\,(t)\;\;[m/s]$\n", fontsize=16)
        plt.xlabel("\n$t\;\;[s]$", fontsize=16)
        plt.ylim([-22, 22])
        plt.tick_params(axis='x', which='major', bottom=True, top=False, labelsize=14)
        plt.tick_params(axis='y', which='major', left=True, right=False, labelsize=14)
        plt.xticks([0, 150, 300, 450, 600], ['$0$', '$150$', '$300$', '$450$', '$600$'])
        plt.yticks([-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25], ['$-25$', '$-20$', '$-15$', '$-10$', '$-5$', '$0$', '$5$', '$10$', '$15$', '$-20$', '$-25$'])
        exec(f'plt.text(400, 16.5, "$u_{{max}} = {round(max(u, key=abs), 2)}\,m/s$\\n$t={int(round(t[np.argmax(u)], 0))}\,s$", fontsize=15, bbox=dict(facecolor="white", alpha=1))')
        plt.plot(t, 0.4*u)
        plt.plot(t[np.argmax(np.abs(u))], max(u, key=abs), 'wo')
        plt.tight_layout()
        exec(f"plt.savefig('./Plotagens/u(t), z = z₂₀, 0 a 600 s.png', dpi=600)")
        #plt.show()
        plt.close()
        print("u(t).png ✓")

    if np.sum(q) != 0:

        plt.figure(8)                   # q(t)
        plt.style.use('classic')
        plt.grid(True)
        plt.ylabel("$q\,(t)\;\;[kN]$\n", fontsize=16)
        plt.xlabel("\n$t\;\;[s]$", fontsize=16)
        plt.ylim([0, 10])
        plt.tick_params(axis='x', which='major', bottom=True, top=False, labelsize=14)
        plt.tick_params(axis='y', which='major', left=True, right=False, labelsize=14)
        plt.xticks([0, 150, 300, 450, 600], ['$0$', '$150$', '$300$', '$450$', '$600$'])
        plt.yticks([0, 2.5, 5, 7.5, 10], ['$0$', '$2,5$', '$5,0$', '$7,5$', '$10$'])
        exec(f'plt.text(400, 7.5, "$q_{{max}} = {round(max(q, key=abs)/1000, 2)}\,kN$\\n$t={int(round(t[np.argmax(q)], 0))}\,s$", fontsize=14, bbox=dict(facecolor="white", alpha=1))')
        plt.plot(t, q/1000)
        plt.plot(t[np.argmax(np.abs(q))], max(q, key=abs)/1000, 'wo')
        plt.tight_layout()
        exec(f"plt.savefig('./Plotagens/q(t), 0 a 600 s.png', dpi=600)")
        #plt.show()
        plt.close()
        print("q(t).png ✓\n")

def salvarTxt(nomeArquivo, vetor):              # Transferência das saídas a um arquivo .txt para aplicação da TRF no Excel
    nome = nomeArquivo + '.txt'
    nomeArquivo = './Saídas txt/' + nomeArquivo + '.txt'
    while True:
        try:
            arquivo2 = open(nomeArquivo,"r+")
            arquivo2.truncate(0)
            arquivo2.close()
            arquivo2 = open(nomeArquivo, "a")
            for j in range(len(t)):
                print(str(vetor[j]).replace('.',','), file=arquivo2)
            arquivo2.close()
            break
        except:
            arquivo2 = open(nomeArquivo, "w")
            for j in range(len(t)):
                print(str(vetor[j]).replace('.',','), file=arquivo2)
            arquivo2.close()
    exec(f'print("{nome} ✓")')

#region
now = datetime.now()
print('\n'+now.strftime("%d/%m/%Y %H:%M:%S"))

# Espectro da velocidade do vento: 0,001 Hz ≤ f ≤ 1,5 Hz
f = np.linspace(0.001,1.5,num=5000)
f_i = 0.005                     # Frequência inicial considerada
f_f = 1.0                       # Frequência final considerada
m = 500                         # Discretização do espectro

freqInfo = [f, f_i, f_f, m]


# Constantes relativas ao fluido e à estrutura
ro = 1.225                  # Densidade do ar ρₐᵣ [kg/m³]
M = 2796.67                 # Massa modal M [kg]
n = 20                      # Discretizações da estrutura
xi = 0.015                  # Razão de amortecimento crítico (ou taxa de amortecimento) ξ [-]
f_n = 0.8459                # Frequência natural da estrutura
omega = f_n*2*np.pi         # Frequência natural angular da estrutura
pondELS = 1                 # Ponderação da carga de vento para ELS


# Constantes relativas ao atenuador pendular
g = 9.80665                             # Aceleração da gravidade
m_p = .1*M                              # Massa do pêndulo [kg]
mu_p = m_p/M                            # Razão entre a massa do pêndulo e da estrutura
xi_p = np.sqrt(3*mu_p/(8*(1+mu_p)**3))  # Taxa de amortecimento do pêndulo
psi_p = 1/(1+mu_p)                      # Razão entre a frequência do pêndulo e da estrutura
f_p = psi_p*f_n                         # Frequência do pêndulo
L_p = g*(m_p+M)**2/(M*omega)**2         # g/(w_p**2)                        # Comprimento do pêndulo (Lₚ = g /ωₚ²)
k_p = 0 #100                            # Rigidez do pêndulo [N/m]
w_p = np.sqrt((k_p/m_p+g*L_p)/(L_p**2)) # 2*np.pi*f_p                       # Frequência angular do pêndulo
c_p = 2*xi_p*m_p*L_p**2*w_p                    # Amortecimento do pêndulo (cₚ = 2 ⋅ ξₚ ⋅ mₚ ⋅ ωₚ)

aten1Info = [m_p, L_p, c_p, g, k_p]


# Forma modal ɸᵢ em cada altura zᵢ
z = [0.0,4.50,9.00,13.50,18.00,22.00,26.00,30.00,33.00,36.00,39.00,42.00,44.00,
46.00,48.00,50.00,52.00,54.00,56.00,58.00,60.00]
phi = [0.0,0.0011,0.0069,0.0178,0.0336,0.0565,0.0836,0.1152,0.1419,0.1714,
0.2036,0.2384,0.2646,0.2949,0.3293,0.3675,0.4111,0.4566,0.5035,0.5509,0.5979]
phi = [i/.5979 for i in phi]
z_i = [2.25,6.75,11.25,15.75,20.00,24.00,28.00,31.50,34.50,37.50,40.50,43.00,45.00,47.00,
49.00,51.00,53.00,55.00,57.00,59.00]
phi_i = interpolate.interp1d(z, phi)(z_i)


# Produto do coeficiente de arrasto Cₐᵢ pela área efetiva Aₑᵢ e velocidade média Uᵢ
Ca_ixAe_i_v90 = [20.237,19.484,18.740,18.004,14.514,14.857,14.656,8.921,8.808,10.049,7.808,4.739,4.546,5.590,4.397,4.532,8.931,4.028,6.135,5.127]
U_i = [18.573,22.511,24.616,26.109,27.223,28.106,28.874,29.476,29.949,30.389,30.801,31.125,31.374,31.614,31.845,32.069,32.285,32.495,32.699,32.897]

estVenInfo = [pondELS, Ca_ixAe_i_v90, phi_i, U_i, ro, M, n, xi, omega]     # Vetor de informações da estrutura e do vento


# Fatores para a definição do vetor de tempo
ti = 0                          # Tempo inicial
tf = 600                        # Tempo final
dt = 0.0366                     # Intervalo entre amostras de tempo
t = vetorDeTempo(ti, tf, dt)


# Cálculo da flutuação do vento
u_n = u_n(t, freqInfo, estVenInfo)


# Condições de contorno para a estrutura livre
d0, v0 = 0, 0
y0_nc = np.zeros(2)
y0_nc[0] = d0
y0_nc[1] = v0


# Condições de contorno para a estrutura controlada
d0, o0, v0, w0 = 0, 0, 0, 0
y0_c = np.zeros(4)
y0_c[0] = d0
y0_c[1] = o0
y0_c[2] = v0
y0_c[3] = w0


# Cálculo da força modal e do coeficiente de amortecimento
q = q(t, estVenInfo, u_n)
alpha = alpha(estVenInfo)


# Resolução da EDO do equilíbrio dinâmico independente para a estrutura não controlada
sol_nc = rk4_1(y0_nc, t, q, alpha, estVenInfo)                # Matriz solução: [d, v]

a1 = sol_nc[0,:]
a_1 = sol_nc[1,:]
a__1 = np.gradient(a_1, dt, edge_order=2)

a1_med = deslocEstat(a1, dt) 
delta1 = (abs(max(a1, key=abs))-a1_med)*100     # Deslocamento dinâmico máximo da estrutura livre

print('\nEstrutura livre')
print('xₘₐₓ =', round(abs(100*max(a1, key=abs)), 2), 'cm, t =', int(round(t[np.argmax(a1)], 0)), 's')
print("ẋₘₐₓ =", round(abs(max(a_1, key=abs)), 2), 'm/s, t =', int(round(t[np.argmax(a_1)], 0)), 's')
print("ẍₘₐₓ =", round(abs(max(a__1, key=abs)), 2), 'm/s², t =', int(round(t[np.argmax(a__1)], 0)), 's')
print('δ₁ =', round(delta1, 2), 'cm')
print('x̄ = ', round(a1_med*100, 1), ' cm', sep='')


# Resolução da EDO do equilíbrio dinâmico independente para a estrutura controlada
sol_c = rk4_2(y0_c, t, q, alpha, estVenInfo, aten1Info)        # Matriz solução: [d_torre, v_ torre, θ_pêndulo, ω_pêndulo]
#sol_c = rk4_3(y0_c, t, q, alpha, estVenInfo, aten2Info)        # Matriz solução: [d_torre, v_ torre, θ_disco, ω_disco]

a2 = sol_c[0,:]
a_2 = sol_c[2,:]
a__2 = np.gradient(a_2, dt, edge_order=2)

o = sol_c[1,:]
o_ = sol_c[3,:]
o__ = np.gradient(o_, dt, edge_order=2)

a2_med = deslocEstat(a2, dt)
delta2 = (abs(max(a2, key=abs))-a2_med)*100        # Deslocamento dinâmico máximo da estrutura controlada
delta3 = 100*L_p*abs(np.tan(max(o, key=abs)))                        # Deslocamento lateral máximo do pêndulo
#delta3 = 100*(max(o, key=abs)-deslocEstat(o, dt))                     # Deslocamento lateral máximo do disco

print('\nEstrutura controlada')
print('xₘₐₓ =', round(abs(100*1.000*max(a2, key=abs)), 2), 'cm, t =', int(round(t[np.argmax(a2)], 0)), 's')
print("ẋₘₐₓ =", round(abs(max(a_2, key=abs)), 2), 'm/s, t =', int(round(t[np.argmax(a_2)], 0)), 's')
print("ẍₘₐₓ =", round(abs(max(a__2, key=abs)), 2), 'm/s², t =', int(round(t[np.argmax(a__2)], 0)), 's')
print('δ₂ =', round(1.000*delta2, 2), 'cm')
print('x̄ = ', round(1.000*a2_med*100, 1), ' cm', sep='')


print('\nSistema de controle')
print('θₘₐₓ =', round(abs(max(o, key=abs)), 2), 'rad, t =', int(round(t[np.argmax(o)], 0)), 's')
print("ωₘₐₓ =", round(abs(max(o_, key=abs)), 2), 'rad/s, t =', int(round(t[np.argmax(o_)], 0)), 's')
print("αₘₐₓ =", round(abs(max(o__, key=abs)), 2), 'rad/s², t =', int(round(t[np.argmax(o__)], 0)), 's')
print('δ₃ =', round(1.000*delta3, 2), 'cm')

print('\nAtenuação')
print('f =', round(f_p, 4), 'Hz')
print('Δx = ', round((1.000*delta1-1.000*delta2), 2), ' cm (eficiência: ', round(100*(1.000*delta1 - 1.000*delta2)/(1.000*delta1), 1), '%)', sep='')
print('L =', round(100*L_p, 1), 'cm\n')

# Plotagem das respostas no tempo
vetores1 = [t, a1, a_1, a__1, np.zeros(len(u_n[-1])), np.zeros(len(q)), dt]
vetores5 = [t, a2, a_2, a__2, o, o_, o__, u_n[-1], q, dt, "z₂₀"]
#salvarFig_nc(vetores1)
#salvarFig_c(vetores5)


# Transferência das saídas para um arquivo .txt
salvarTxt('deslocamento_nc', a1)
salvarTxt('velocidade_nc', a_1)
salvarTxt('aceleração_nc', a__1)
salvarTxt('aceleração_c', a__2)
salvarTxt('deslocamento_c', a2)
salvarTxt('velocidade_c', a_2)
salvarTxt('deslocamento_at', o)
salvarTxt('flutuação z1', u_n[0])
salvarTxt('flutuação z2', u_n[1])
salvarTxt('flutuação z3', u_n[2])
salvarTxt('flutuação z4', u_n[3])
salvarTxt('flutuação z5', u_n[4])
salvarTxt('flutuação z6', u_n[5])
salvarTxt('flutuação z7', u_n[6])
salvarTxt('flutuação z8', u_n[7])
salvarTxt('flutuação z9', u_n[8])
salvarTxt('flutuação z10', u_n[9])
salvarTxt('flutuação z11', u_n[10])
salvarTxt('flutuação z12', u_n[11])
salvarTxt('flutuação z13', u_n[12])
salvarTxt('flutuação z14', u_n[13])
salvarTxt('flutuação z15', u_n[14])
salvarTxt('flutuação z16', u_n[15])
salvarTxt('flutuação z17', u_n[16])
salvarTxt('flutuação z18', u_n[17])
salvarTxt('flutuação z19', u_n[18])
salvarTxt('flutuação z20', u_n[-1])
salvarTxt('força_modal', q)
#endregion

# Fim do processamento
now = datetime.now()
print('')
print(now.strftime("%d/%m/%Y %H:%M:%S"))
print("\nTempo de processamento: %s segundos\n" % (int(time.time() - start_time)))