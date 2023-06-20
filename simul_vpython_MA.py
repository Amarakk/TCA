import numpy as np
from sympy import symbols, expand, Matrix, Symbol
import sympy as sp
import matplotlib.pyplot as plt
from scipy.linalg import solve
from vpython import *
import math

def dados_simul():
    RK4 = lambda f : lambda x, u, dt :(lambda dx1:(lambda dx2:(lambda dx3:(lambda dx4:(dx1+2*dx2+2*dx3+dx4)/6)(dt*f(x+dx3,u)))(dt*f(x+dx2/2,u)))(dt*f(x+dx1/2,u)))(dt*f(x,u))
    dx = RK4(lambda x, u: A@x + B@u)
    dx_est = RK4(lambda x_est, u: A@x_est + B@u + L@(y-y_est))
    
    m1=1.0; m2=1.0; m3=1.0; k1=5.0; k2=5.0; k3=5.0; b1=0.5; b2=0.5; b3=0.5
    A = np.array([
        [0,0,0,1,0,0],
        [0,0,0,0,1,0],
        [0,0,0,0,0,1],
        [-(k1+k2)/m1, k2/m1, 0, -(b1+b2)/m1, b2/m1, 0],
        [0, -(k2+k3)/m2, k3/m2, 0, -(b2+b3)/m2, b3/m2],
        [0, k3/m3, -k3/m3, 0, b3/m3, -b3/m3]
    ])
    B = np.array([
        [0],
        [0],
        [0],
        [0],
        [1/m2],
        [0]
    ])
    t0, tf, dt = 0, 20, .01
    x0, u0 = np.array([[0.5],[1],[1.5],[0],[0],[0]]), np.vstack(np.zeros(B.shape[1]))

    T, X, U = t, x, u = t0, x0, u0


    open_loop_poles = np.roots(np.poly(A))

    print("Polos de malha aberta:")
    print(open_loop_poles)
    print('\n')

    plt.scatter(open_loop_poles.real, open_loop_poles.imag, marker='x', color='red')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.axvline(x=0, color='black', linestyle='--')
    plt.xlabel('Parte Real')
    plt.ylabel('Parte Imaginária')
    plt.title('Polos de Malha Aberta')
    plt.grid(True)
    plt.show()

 
    # Desse comentario até a marcação ta tudo um caos, mas é a parte de adicionar o controlador, se comentar roda sem controle tranquilo.
    # ele até roda mas aqui pra mim ele fica muito tempo no cálculo do determinante, dai n sei se ta errado ou se meu pc é ruim mesmo
    
    # Calcula os polos da malha aberta
    poles, _ = np.linalg.eig(A)

    # Mova os polos 2 unidades para a esquerda
    shifted_poles = [-2 for x in range(0,6)]
    print("Polos desejados:")
    print(shifted_poles)
    print('\n')


    # Variáveis simbólicas
    s = symbols('s')

    # Constrói o polinômio característico com os polos desejados
    char_poly = expand((s-shifted_poles[0])*(s-shifted_poles[1])*(s-shifted_poles[2])*(s-shifted_poles[3])*(s-shifted_poles[4])*(s-shifted_poles[5]))

    print("Polinômio característico:")
    print(char_poly)
    print('\n')
    print(Matrix(A))
    print('\n')
    print(Matrix(B))
    print('\n')
    print(Matrix(symbols('k1:7')).reshape(1,6))
    print('\n')
    print(Matrix(s * np.eye(6)))
    print('\n')

    # Construindo a matriz sI - A + BK
    print((np.dot(B,Matrix(symbols('k1:7')).reshape(1,6))))
 
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)



    k1,k2,k3,k4,k5,k6 = sp.symbols(['k1','k2','k3','k4','k5','k6'])
    K=[[k1,k2,k3,k4,k5,k6]]

    print('\n')
    print(sp.Matrix(s*np.identity(6)-A+np.dot(B,K)))
    print('\n')
    print(sp.collect(sp.Matrix(s*np.identity(6)-A+np.dot(B,K)).det(),s))
    print('\n')
    # 



    kMatrix = Matrix([
        [0, 0, 0, 0, 1, 0, 9.5],
        [0, 1, 0, 0.5, 1.5, 0.5, 33.25],
        [0.5, 1.5, 0.5, 5.25, 15.5, 5.5, 124.75],
        [5.25, 15.5, 5.5, 5, 10, 10, 57.5],
        [5, 10, 10, 25, 50, 50, 117],
        [25, 50, 50, 0, 0, 0, -186],
        ]).rref()

    print(kMatrix)
    print('\n')
    K =  np.array([[(-41256)/625, 13259/625, 5044/625, 2484/625, 19/2,(-5252)/625]])

    # K = [x[-1] for x in kMatrix[0]]
    # K = np.array([[round(x,4) for x in kMatrix[0].col(-1)]])
    # print(K1)
    print(K)
    print('\n')

    

    C = np.array([0,1,0,0,0,0])

    # (100*k1+40*s^6+40*k4*s^5+64*s^5+40*k1*s^4+24*k4*s^4+865*s^4+24*k1*s^3+241*k4*s^3+621*s^3+241*k1*s^2+20*k4*s^2+3735*s^2+20*k1*s+100*k4*s+400*s+1500)/40

    #  https://matrixcalc.org/pt/det.html#determinant-Gauss%28%7B%7Bs,0,0,-1,0,0%7D,%7B0,s,0,0,-1,0%7D,%7B0,0,s,0,0,-1%7D,%7Ba+15,b-5,c,d+s+1,f-0%2e5,g%7D,%7B0,5,-2%2e5,0,s+0%2e5,-0%2e25%7D,%7B0,-1,1,0,-0%2e1,s+0%2e1%7D%7D%29


    # --------------------------------------------------------------//------------------------------------------------------------------

    for i in range(int((tf-t)/dt)):
        # u = (N@r-K@x_est)
        t, x = t + dt, x + dx(x,u,dt)
        u = (-K@x)
        # y, y_est = C@x, C@x_est
        # x_est = x_est + dx_est(x_est,u,dt) # estimação do estado
        X, U, T = np.append(X,x,axis=1), np.append(U,u,axis=1), np.append(T,t)


    f1, f2, f3 = U[0]*0, U[0], U[0]*0
    return T, X, f1, f2, f3

def imprime():
    plt.plot(T, X[0], 'k')
    plt.xlabel('tempo (s)')
    plt.ylabel('x1 (m)')
    plt.grid(True)
    plt.show()

    plt.plot(T, X[1], 'k')
    plt.xlabel('tempo (s)')
    plt.ylabel('x2 (m)')
    plt.grid(True)
    plt.show()

    plt.plot(T, X[2], 'k')
    plt.xlabel('tempo (s)')
    plt.ylabel('x3 (m)')
    plt.grid(True)
    plt.show()

def inicializa():
    tam_eixo, tam_cubo, tam_mola, esp_chao = 2, 2, 5, .05
    cena1 = canvas(title='Simulação massa-mola-amortecedor com controle', width=640, height=300, center=vector(8,0,0), background=color.white)
    dir1 = vector(1,0,0)
    dir2 = -dir1
    forca1 = arrow(pos=vector(0,tam_cubo,0), axis=dir1, color=color.green)
    forca2 = arrow(pos=vector(0,tam_cubo,0), axis=dir1, color=color.red)
    forca3 = arrow(pos=vector(0,tam_cubo,0), axis=dir1, color=color.blue)
    mola1 = helix(vector=dir, thickness=.2, color=color.blue)
    mola2 = helix(vector=dir, thickness=.2, color=color.blue)
    mola3 = helix(vector=dir, thickness=.2, color=color.blue)
    arrow(axis=vector(tam_eixo,0,0), color=color.red), arrow(axis=vector(0,tam_eixo,0), color=color.green), arrow(axis=vector(0,0,tam_eixo), color=color.blue)
    massa1 = box(opacity=.5, size=2*tam_cubo*vec(1,1,1), color=color.green)
    massa2 = box(opacity=.5, size=2*tam_cubo*vec(1,1,1), color=color.red)
    massa3 = box(opacity=.5, size=2*tam_cubo*vec(1,1,1), color=color.blue)
    chao = box(pos=vec(15,-(tam_cubo+esp_chao),0), size=vec(30,2*esp_chao,2*tam_cubo), color=vec(.8, .8, .8))
    graf1 = graph(title='Posição', width=600, height=300, xtitle='<i>t</i> (s)', ytitle='<i>x</i><sub>1</sub> (m)    <i>x</i><sub>2</sub> (m)    <i>x</i><sub>3</sub> (m)',
                  fast=True, xmin=T.min(), xmax=T.max())
    graf2 = graph(title='Força', width=600, height=300, xtitle='<i>t</i> (s)', ytitle='<i>F</i><sub>1</sub> (N)  <i>F</i><sub>2</sub> (N)  <i>F</i><sub>3</sub> (N)',
                  fast=True, xmin=T.min(), xmax=T.max())
    graf3 = graph(title='Velocidade', width=600, height=300, xtitle='<i>t</i> (s)', ytitle='<i>v</i><sub>1</sub> (m/s)    <i>v</i><sub>2</sub> (m/s)    <i>v</i><sub>3</sub> (m/s)',
                  fast=True, xmin=T.min(), xmax=T.max())
    return (forca1, forca2, forca3, mola1, mola2, mola3, massa1, massa2, massa3,
            gcurve(graph=graf1, color=color.green), gcurve(graph=graf1, color=color.red), gcurve(graph=graf1, color=color.blue),
            gcurve(graph=graf2, color=color.green), gcurve(graph=graf2, color=color.red), gcurve(graph=graf2, color=color.blue),
            gcurve(graph=graf3, color=color.green), gcurve(graph=graf3, color=color.red), gcurve(graph=graf3, color=color.blue),
            tam_cubo, tam_mola)

def move():
    delta_f = lambda x: 0 if x < 0 else 2*tam_cubo
    gx1.delete(), gx2.delete(), gx3.delete(), gf1.delete(), gf2.delete(), gf3.delete(), gv1.delete(), gv2.delete(), gv3.delete()
    x1, x2, x3, v1, v2, v3 = X[0]+tam_mola, X[1]+2*(tam_mola+tam_cubo), X[2]+3*(tam_mola+tam_cubo), X[3], X[4], X[5]
    disp_rate = 1/(T[1]-T[0])

    for i in range(len(T)):
        rate(disp_rate)
        mola1.axis.x = x1[i]
        massa1.pos.x = x1[i]+tam_cubo
        mola2.pos.x, mola2.axis.x = x1[i]+2*tam_cubo, x2[i]-x1[i]-2*tam_cubo
        massa2.pos.x = x2[i]+tam_cubo
        mola3.pos.x, mola3.axis.x = x2[i]+2*tam_cubo, x3[i]-x2[i]-2*tam_cubo
        massa3.pos.x = x3[i]+tam_cubo
        forca1.pos.x, forca1.axis.x = x1[i]+delta_f(f1[i]), f1[i]/2
        forca2.pos.x, forca2.axis.x = x2[i]+delta_f(f2[i]), f2[i]/2
        forca3.pos.x, forca3.axis.x = x3[i]+delta_f(f3[i]), f3[i]/2
        gx1.plot(T[i], X[0][i])
        gx2.plot(T[i], X[1][i])
        gx3.plot(T[i], X[2][i])
        gv1.plot(T[i], X[3][i])
        gv2.plot(T[i], X[4][i])
        gv3.plot(T[i], X[5][i])
        gf1.plot(T[i], f1[i])
        gf2.plot(T[i], f2[i])
        gf3.plot(T[i], f3[i])

T, X, f1, f2, f3 = dados_simul()
print(f1, f2, f3)
imprime()
forca1, forca2, forca3, mola1, mola2, mola3, massa1, massa2, massa3, gx1, gx2, gx3, gf1, gf2, gf3, gv1, gv2, gv3, tam_cubo, tam_mola = inicializa()
move()