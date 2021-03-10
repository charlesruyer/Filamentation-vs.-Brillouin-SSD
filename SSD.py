#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 15:51:42 2020

@author: Charles
"""

from scipy.linalg import expm, inv

import scipy.special
from scipy import signal
from scipy.special import exp1
from scipy.special import  jv
import numpy as np
import numpy.matlib as ma
from scipy.interpolate import interp1d
import os.path, glob, re
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import cm
from scipy.integrate import odeint
from scipy.integrate import complex_ode
from scipy.integrate import ode
from scipy.special import sici
from scipy.special import  lambertw
from scipy.special import  erf
from scipy.special import  erfi
from scipy import integrate
from scipy import optimize 
import pylab
import sys


# Genere les champs a la focal (2d), E(y,t) d'un faisceau ssd
def Field_ssd(y,t,f=8,N=100,k0=2*np.pi/0.35e-6,wm=2*np.pi*14.25e9,m=15,figure=1,env=True):
    c=3e8
    if env:
        w0=0
    else:
        w0=k0*c
    km=k0/(2*f)
    Tr=2*np.pi/wm
    Y,T = np.meshgrid(y,t)
    E=0*Y
    
    kp   = np.linspace(-km,km,100)
    rand = np.random.uniform(0,1,N)
    phi  =  (rand>0.5) * np.pi
    for n in range(N):
        for l in range(-m,m+1,1):
            w=w0+l*wm
            t0 = Tr*(kp[n]+km)/(2*km)
            E += jv(l,m)*np.cos(kp[n]*Y  -w*(T-t0) +  phi[n] )/N
            
        
    if figure is None:
        return E
    else:
        E=E.T
        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure+1,figsize=[7,5])
        fig.clf()
        ax0 = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.22, right=0.95, top=0.9, bottom=0.2)
        #cf0 = plt.pcolor(t*1e12 ,y*1e6 , E**2 ,cmap =  plt.cm.RdBu, vmin =0, vmax =1 )
        cf0 = plt.pcolor(t*1e12 ,y*1e6 , E**2 ,cmap =  plt.cm.gist_earth_r, vmin =0, vmax =np.max(E**2) )
        plt.colorbar(cf0)#, ticks=[ -1,0, 1])
        #ax0.set_xlim(-0.01,0.01)
        #ax0.set_ylim(-0.01,0.01)
        #plt.axes().set_aspect('equal', 'datalim')
        ax0.set_xlabel("$t$ (ps)")
        ax0.set_ylabel("$y$ ($\mu$m)")
        fig.canvas.draw()
    
# Fonction aui apparait dans la dispe filam Fsbs RPP+SSD
def plot_jv2d(l1=np.linspace(-5,5,100),m=5,l2=np.linspace(-5,5,101),figure=1):
    L1,L2=np.meshgrid(l1,l2)
    p=jv(L1,m)*jv(L2,m)
    if figure is None:
        return p
    else:
        p=p
        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure+1,figsize=[7,5])
        fig.clf()
        ax0 = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.22, right=0.95, top=0.9, bottom=0.2)
        cf0 = plt.pcolor(l1,l2,p ,cmap =  plt.cm.RdBu)#, vmin =0, vmax =np.max(E**2) )
        plt.colorbar(cf0)#, ticks=[ -1,0, 1])
        #ax0.set_xlim(-0.01,0.01)
        #ax0.set_ylim(-0.01,0.01)
        #plt.axes().set_aspect('equal', 'datalim')
        ax0.set_xlabel("$l_1$")
        ax0.set_ylabel("$l_2$")
        fig.canvas.draw()
    
def taux_collisionnel(masse=None,charge=None,dens=None,temp=None,vd=None,*aqrgs,**kwargs):
    #% Frequences de collision de Spitzer
    #% Ref. A. Decoster, Modeling of collisions (1998)
    #%
    #% Input : [beam, background]
    #% . masse(1:2)  : masse/me
    #% . charge(1:2) : charge/qe
    #% . dens(1:2)   : densite (cm^-3)
    #% . temp(1:2)   : temperature (eV)
    #% . vde(1:2)    : vitesses de derive (cm/s)
    #% Output :
    #% . nu_imp  : momentum transfer frequency (1/s)
    #% . nu_ener : Energy transfer frequency (1/s)
    #% . lambda  : mean-free-path (cm)
    masse = np.array(masse)
    charge = np.array(charge)
    dens = np.array(dens)
    temp = np.array(temp)
    vd = np.array(vd)
    
    #varargin = cellarray(args)
    #nargin = 5-[masse,charge,dens,temp,vd].count(None)+len(args)
    
    vt1=30000000000.0 * (temp[0] / (masse[0] * 511000.0)) ** 0.5
    vd1=np.abs(vd[0])
    if all(masse == 1):
        ##print('All electrons'
        if temp[1] < 10:
            log_coul=23 - np.log(dens[1] ** 0.5 * temp[1] ** (- 1.5))
        else:
            log_coul=24 - np.log(dens[1] ** 0.5 * temp[1] ** (- 1))
        ##print(log_coul, temp
    else:
        if any(masse == 1.):
            ##print('electron-ion'
            if masse[0] ==1.:
                ielec=0
                ##print('indice electron: ',ielec
                iion=1
                ##print('indice ion: ',iion
            else:
                ielec=1
                ##print('indice electron: ',ielec
                iion=0
                ##print('indice ion: ',iion
            
            
            if (temp[iion] / masse[iion] < temp[ielec]) and (temp[ielec] < 10 * charge[iion] ** 2):
                log_coul=23 - np.log(dens[ielec] ** 0.5 * charge[iion] * temp[ielec] ** (- 1.5))
            else:
                if 10 * charge[iion] ** 2 < temp[ielec]:
                    log_coul=24 - np.log(dens[ielec] ** 0.5 * temp[ielec] ** (- 1))
                else:
                    if temp[ielec] < temp[iion] * charge[iion] / masse[iion]:
                        mu=masse[iion]/1836.
                        log_coul=30 - np.log(dens[iion] ** 0.5 * temp[iion] **(-1.5) * charge[iion] ** 2 / mu)
                    else:
                        print( 'No Coulombien logarithm from Lee and Moore')
                        return  {"nup":None,"nuk":None,"log_coul":None,"lmfp":None}
            	##print('Log Coulombien: ',log_coul
        else:
            log_coul=23 - np.log(charge[0] * charge[1] * (masse[0] + masse[1]) * (dens[0] * charge[0] ** 2 / temp[0] + dens[1] * charge[1] ** 2 / temp[1]) ** 0.5 / (masse[0] * temp[1] + masse[1] * temp[0]))

    qe=4.8032e-10
    temp=1.6022e-19 * 10000000.0 * temp
    masse=9.1094e-28 * masse
    m12=masse[0] * masse[1] / (masse[0] + masse[1])
    nu_imp=(4. / 3.) * (2 * np.pi) ** 0.5 * dens[1] * charge[0] ** 2 * charge[1] ** 2 * qe ** 4 * log_coul / (masse[0] * m12 * (temp[0] / masse[0] + temp[1] / masse[1] + (vd[0] - vd[1]) ** 2) ** 1.5)
    nu_ener=(8. / 3.) * (2 * np.pi) ** 0.5 * dens[1] * charge[0] ** 2 * charge[1] ** 2 * qe ** 4 * log_coul / (masse[0] * masse[1] * (temp[0] / masse[0] + temp[1] / masse[1] + (vd[0] - vd[1]) ** 2) ** 1.5)

    _lambda=np.max([vt1,vd1]) / nu_imp
    ##print('nu_imp = ',nu_imp,' Hz'
    ##print('nu_ener = ',nu_ener,' Hz'
    ##print('tau_imp = ',1/nu_imp,' s'
    ##print('tau_ener = ',1/nu_ener,' s', dens, log_coul,masse
    ##print('log_coul = ',log_coul
    ##print('Mean-free-path: ',_lambda,' cm'
    result = {"nup":nu_imp,"nuk":nu_ener,"log_coul":log_coul,"lmfp":_lambda}
    return result


#Fonction plasma et ses derive a variable REELLE !
def Z_scalar(x):
    if np.abs(x)<=25:
        res = np.pi**0.5*np.exp(-x**2)*( 1j - erfi(x) ) 
    else:
        res=-1/x -1./x**3/2. +1j*np.sqrt(np.pi)*np.exp(-x**2)
    return res
def Z(x):
    if np.size(x)==1:
        return Z_scalar(x)
    else:
        res = 0*x+0j*x
        for i in range(len(x)):
            ##print('i= ',i, x[i], Z_scalar(x[i])
            res[i] = Z_scalar(x[i]) 
        return res 

def Zp(x):
    return -2*(1 + x*Z(x)) 

def Zp_scalar(x):
    return -2*(1 + x*Z_scalar(x))

def Zpp(x):
    return  -2*(Z(x) + x*Zp(x))

def Zppp(x):
    return  -2*( 2*Zp(x) + x*Zpp(x) )

def d_betak(xi,xe, Z,A,Te,Ti):
    ve2 = np.sqrt(2*Te/511000.)*3e8
    vi2 = np.sqrt(2*Ti/511000./1836./A)*3e8
    ZTesTi = Z*Te/Ti
    res  = -0.5*Zpp(xe)*Zp(xi)/(Zp(xi)+Zp(xe)/ZTesTi)/ve2
    res += -0.5*Zp(xe) *(Zpp(xi)/vi2*(Zpp(xi)+Zpp(xe)/ZTesTi) - Zp(xi)*(Zpp(xi)/vi2+Zpp(xe)/ZTesTi/ve2) )/ (Zp(xi)+Zpp(xe)/ZTesTi)**2 
    return res

# Fonction de transfert cinetique de la force ponderomotrice
# Zp(xie)/2 * ( 1 - sum_i Zp(xi) ) /eps
def alpha_kin(Te, Ti, Z, A, nisne, vphi, k2lde2=0, ne=1,figure=None, is_chiperp = False):
    c=3e8
    if ne ==0 : # On neglige les e-
        Zpxie = -2+0j
    else: 
        xie = vphi /np.sqrt(2*Te/511000. )/c
        Zpxie = Zp(xie)
    ##print('Zp(xie) = ', Zpxie
    Xe= Zpxie * 1.
    sumXi = 0. + 0j
    for i in range(len(Ti)):
        xi = vphi /np.sqrt(2*Ti[i]/511000./1836/A[i] )/c
        sumXi += Zp(xi) * nisne[i] *Te* Z[i]**2/Ti[i]
        
        ##print('Zp(xii) = ', Zp(xi)
    # Si k2lde2 ==0  on suppose k2lde2 <<  1 (Longeur de Debaye electronique
    ak = -0.5*Zpxie *(k2lde2- sumXi) / (k2lde2 - Xe-sumXi)
    ##print(np.shape(np.real(ak)), np.shape(np.imag(ak))

    fluctuation=np.imag(1./ (+Xe+sumXi))
    if figure is None:
        return  ak
    else:
        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure, figsize=[8,5])
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.2)        
        a, = ax.plot(vphi/c,np.real(ak),'k', linewidth=2)              
        a, = ax.plot(vphi/c,np.imag(ak),'--k', linewidth=2)              
        ax.set_ylabel("$F_\mathrm{kin}$")
        ax.set_xlabel("$v_\phi/c$ ")
        #if ylog:
        #    ax.set_yscale("log")
        #if xlog:
        #    ax.set_xscale("log")
        #ax.set_xlim(0,np.max(kydose/k0))
        fig.canvas.draw()
        plt.show()

        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figure+1, figsize=[8,5])
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.2)        
        a, = ax.plot(vphi/c,fluctuation,'k', linewidth=2)                
        ax.set_ylabel("$\\Im( 1./\\epsilon)$")
        ax.set_xlabel("$v_\phi/c$ ")
        #if ylog:
        #    ax.set_yscale("log")
        #if xlog:
        #    ax.set_xscale("log")
        #ax.set_xlim(0,np.max(kydose/k0))
        fig.canvas.draw()
        plt.show()
        

def dispe_filam_kinSSD(Te=1.e3,Ti=[300.],Z=[1.],A=[1.],nesnc=0.1,I0=3e14,f=8.,wm=2*np.pi*14.25e9,Tr=70e-12,m=15,k0=2*np.pi/0.35e-6,nisne=None,isnonlocal=True,k=None,vpscs=None,figure=1,gmax=None,kmax=None,gticks=None,kticks=None):
    #dispe_filam_kinRPP(Te=700.,Ti=[500,500.],Z=[1.,6.],A=[1.,12.],nisne=[1./7.,1./7.],gmax=3.e-2,kmax=0.1,gticks=[0,0.01,0.02,0.03,0.04,0.05],kticks=[-0.1,0,0.1])
    #dispe_filam_kinRPP(Te=1000.,Ti=[300],Z=[1.],A=[1.],gmax=8e-3,kmax=0.1,gticks=[0,0.002,0.004,0.006,0.008],kticks=[-0.1,0,0.1])
    c=3e8
    km=k0/(2*f)
    nc=k0**2*c**2*8.85e-12*9.1e-31/(1.6e-19)**2
    print('nc = ',nc, ' m^-3')
    if nisne is None:
        nisne = [1./np.float(np.array(Z))]
        print('nisne = ',nisne)
    dnsn = I0*1e4/(nc*c*Te*1.6e-19)
    print('dnsn = ',dnsn)
    n=np.sqrt(1-nesnc)
    w0=k0*c/n
    Tim, Am, Zm = np.mean(np.array(Ti)),np.mean(np.array(A)),np.mean(np.array(Z))
    cs = np.sqrt((Zm*Te+3*Tim)/(1836.*511000.*Am ))*c
    zeta= Zm*Te/511000./(1836*Am*cs**2/c**2)

    mm=0
    for i in range(len(Ti)):
        mm+=Z[i]**2*nisne[i]*Te/Ti[i]
    alphak = m/(1.+m)
    
    ne=nesnc*nc/1e6
    print([1.,A[0]*1836.],[-1.,Z[0]],[ne,ne/Z[0]],[Te,Ti[0]])
    r=taux_collisionnel(masse=[1.,A[0]*1836.],charge=[-1.,Z[0]],dens=[ne,ne*nisne[0]],temp=[Te,Ti[0]],vd=[0,0])
    print(r['lmfp'])
    lmfp=r['lmfp']*1e-2
    iak=0
    if len(Z)>1:
        for i in range(1,len(Z)):
            r=taux_collisionnel(masse=[1.,A[i]*1836.],charge=[-1.,Z[i]],dens=[ne,ne*nisne[i]],temp=[Te,Ti[i]],vd=[0,0])
            print('lmfp, i = ',i, '   : ',r['lmfp'])
            if lmfp  > r['lmfp']*1e-2:
                lmfp = r['lmfp']*1e-2 
                iak  = i

    print('lmfp = ',lmfp,' m')

    #Brillouin avant
    if k is None:
        k = np.linspace(0.03*k0*dnsn**0.5,3*k0*dnsn**0.5,100)
    x=lmfp*Z[i]**0.5*k
    Ak =( 0.5 +isnonlocal*Z[i]*(0.074/x**2 +0.88/x**(4./7.) +2.54/(1.+5.5*x**2)) )

    #g0= np.sqrt(np.pi) *(cs/c)**3*(np.exp(-Am*1836*9.1e-31*cs**2/(2*Tim*1.6e-19))/(2*Tim/511000.)**1.5 +Am*1836./Zm/(2*Te/511000.)**1.5)
    vi=np.sqrt(Tim/(Am*1836*511000.))*c
    g0= np.sqrt(np.pi/8) *(cs/vi)**3*(np.exp(-0.5*(cs/vi)**2 ) +np.sqrt(Zm/(Am*1836.)) *(Zm*Te/Tim)**-1.5)

    print('g_0 = ', g0)

    if vpscs is None:
        vphi = np.linspace(0.8*cs, 1.2*cs,100)
    else:
        vphi=vpscs*cs
    V, K=np.meshgrid(vphi,k)
    alphakold=alphak
    print('alpha[0] = ',alphak )
    alphak = alpha_kin(Te,Ti, Z, A, nisne, (vphi), k2lde2=0, ne=1)
    alphaf =  zeta/( 1-2*1j*g0*vphi/cs - (vphi/cs)**2  )
    print('alpha[0] = ',alphak[0] )
    print(np.shape(alphak), np.shape(Ak))
    #print('alphak = ',alphak)
    
    
    N=np.linspace(-m,m,int(2*m+1))
    SJ = np.sum(jv(n,m)**2)
    print('Sum Jv^2 = ',SJ)
    wlsw0 = 1+N*wm/w0
    print('Tr = ',Tr*1e12 , ' ps')
    print('wm, m = ',wm,m)
   
    U = 0*V +0j*V
    Urpp = 0*V +0j*V
    #Um = 0*V +0j*V
    #Ufm = 0*V +0j*V
    for iv in range(len(vphi)):
        print('vphi : ',iv,' / ',len(vphi))
        for ik in range(len(k)): 
            #cinetique RPP+SSD
            betak = 0
            for l1 in range(-m,m+1):
                for l2 in range(-m,m+1):
                    vpt = (vphi[iv]*k[ik] + wm*(l1-l2))/np.sqrt(k[ik]**2 + wm**2/w0**2*k0**2*(l1-l2)**2)
                    alphak = alpha_kin(Te,Ti, Z, A, nisne, vpt, k2lde2=0, ne=1)
                    arg=np.pi*0.5*Tr*wm*(l1-l2)
                    betak +=jv(l1,m)*jv(l2,m)*np.sinc(arg/np.pi)*np.exp(1j*arg)*alphak
            B = - (4*km*k[ik]*c**2 )/w0**2*(1. /( nesnc*dnsn/n**2*betak*Ak[ik])) 
            exp = np.exp( B)
            if np.real(B)>1e20:
                print('-1/B = ', B)
                E=  ( (k[ik]/k0+1./f )**2  )
            if np.abs(B)<1e-10:
                E=  ( (k[ik]/k0-1./f)- (1+ B)*(k[ik]/k0+1./f)**2  ) /(- B)
            else:
                exp = np.exp(  B)
                #print(B,np.exp(B))
                E=  ( (k[ik]/k0-1./f) - exp*(k[ik]/k0+1./f ) ) /(1-exp)
            U[ik,iv] = vphi[iv]/c/n+0.5*E
            
            #cinetique Rpp
            betak = 0
            for l1 in range(-m,m+1):
                for l2 in range(-m,m+1):
                    vpt = (vphi[iv]*k[ik] + 0*wm*(l1-l2))/np.sqrt(k[ik]**2 + 0*wm**2/w0**2*k0**2*(l1-l2)**2)
                    alphak = alpha_kin(Te,Ti, Z, A, nisne, vpt, k2lde2=0, ne=1)
                    arg=0
                    betak +=jv(l1,m)*jv(l2,m)*np.sinc(arg/np.pi)*np.exp(1j*arg)*alphak
            B = - (4*km*k[ik]*c**2 )/w0**2*(1. /( nesnc*dnsn/n**2*betak*Ak[ik])) 
            exp = np.exp( B)
            if np.real(B)>1e20:
                print('-1/B = ', B)
                E=  ( (k[ik]/k0+1./f )**2  )
            if np.abs(B)<1e-10:
                E=  ( (k[ik]/k0-1./f)- (1+ B)*(k[ik]/k0+1./f)**2  ) /(- B)
            else:
                exp = np.exp(  B)
                #print(B,np.exp(B))
                E=  ( (k[ik]/k0-1./f) - exp*(k[ik]/k0+1./f ) ) /(1-exp)
            Urpp[ik,iv] = vphi[iv]/c/n+0.5*E
            
            
    #U= Up*(np.imag(Up)>0) + Um*(np.imag(Um)>0)
    #Uf= Ufp*(np.imag(Ufp)>0) + Ufm*(np.imag(Ufm)>0)
    ###############
    #U= Up*(np.imag(Up)<0) + Um*(np.imag(Um)<0)
    #Uf= Ufp*(np.imag(Ufp)<0) +Ufm*(np.imag(Ufm)<0)
            
    gamma = -K*np.imag(U)
    kpara = K*np.real(U)
    gammarpp = -K*np.imag(Urpp)
    kpararpp = K*np.real(Urpp)
    
    if np.sum(np.abs(gamma))== 0 :
        print('No instability with ssd !')
    else:
        print('Max Gamma/k0 = ' , np.max(gamma)/k0)
    
    #np.arctan2(np.real(U2),np.imag(U2))/2.
    #uabs = np.sqrt(np.abs(U2))
    #gamma = K*uabs*np.cos(uth)
    #kpara = -K*uabs*np.sin(uth)

    #mesh  = gamma>0
    #gamma *= mesh
    #kpara *= mesh
    for iv in range(len(vphi)):
        for ik in range(len(k)):
            if np.isnan(gamma[ik,iv]) or np.isnan(kpara[ik,iv]):
                gamma[ik,:iv] = 0
                kpara[ik,:iv] = 0

            if np.isnan(gammarpp[ik,iv]) or np.isnan(kpararpp[ik,iv]):
                gammarpp[ik,:iv] = 0
                kpararpp[ik,:iv] = 0


    #gmax=np.max([np.max(gamma),np.max(gammaf)])/k0
    #kmax=np.max([np.max(kpara),np.max(kparaf)])/k0
    #print('gamma/k0 = ',gamma /k0
    #print('kpara/k0 = ',kpara/k0
    if figure is not None:
       

        plt.rcParams.update({'font.size': 25})
        fig = plt.figure(figure+1,figsize=[8,6])
        fig.clf()
        ax0 = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.18, right=0.9, top=0.9, bottom=0.2)
        data=gamma.T/k0
        #data = np.log10(gamma)
        if gmax is not None:  
            cf0 = plt.pcolor(k/k0,vphi/cs,data,cmap=plt.cm.gist_earth_r,vmin =0, vmax =gmax)
        else:
            cf0 = plt.pcolor(k/k0,vphi/cs,data,cmap=plt.cm.gist_earth_r)
        if gticks is not None:
            plt.colorbar(cf0, ticks=gticks)
        else:
            plt.colorbar(cf0)
        ax0.set_xlabel("$\\vert k_s\\vert /k_0$")
        ax0.set_ylabel("$v_\\phi/c_s$")
        #ax0.set_xlim(np.min(ztesti),np.max(ztesti))
        #ax0.set_ylim(np.min(M),np.max(M))
        #if taussd[1]-taussd[0] !=taussd[2]-taussd[1]:
        #    print('xscale'
        #    ax0.set_xscale('log')
        fig.canvas.draw()
        plt.show()

        plt.rcParams.update({'font.size': 25})
        fig = plt.figure(figure+2,figsize=[8,6])
        fig.clf()
        ax0 = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.18, right=0.9, top=0.9, bottom=0.2)
        data=-kpara.T/k0
        #data = np.log10(gamma)
        if kmax is not None:  
            cf0 = plt.pcolor(k/k0,vphi/cs,data,cmap=plt.cm.gist_earth_r,vmin =-kmax, vmax =kmax)
        else:
            cf0 = plt.pcolor(k/k0,vphi/cs,data,cmap=plt.cm.gist_earth_r)
        if kticks is not None:
            plt.colorbar(cf0, ticks=kticks)
        else:
            plt.colorbar(cf0)
        ax0.set_xlabel("$\\vert k_s\\vert /k_0$")
        ax0.set_ylabel("$v_\\phi/c_s$")
        #ax0.set_xlim(np.min(ztesti),np.max(ztesti))
        #ax0.set_ylim(np.min(M),np.max(M))
        #if taussd[1]-taussd[0] !=taussd[2]-taussd[1]:
        #    print('xscale'
        #    ax0.set_xscale('log')
        fig.canvas.draw()
        plt.show()

        plt.rcParams.update({'font.size': 25})
        fig = plt.figure(figure+3,figsize=[8,6])
        fig.clf()
        ax0 = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.18, right=0.9, top=0.9, bottom=0.2)
        data=gammarpp.T/k0
        #data = np.log10(gamma)
        if gmax is not None:  
            cf0 = plt.pcolor(k/k0,vphi/cs,data,cmap=plt.cm.gist_earth_r,vmin =0, vmax =gmax)
        else:
            cf0 = plt.pcolor(k/k0,vphi/cs,data,cmap=plt.cm.gist_earth_r)
        if gticks is not None:
            plt.colorbar(cf0, ticks=gticks)
        else:
            plt.colorbar(cf0)
        ax0.set_xlabel("$\\vert k_s\\vert /k_0$")
        ax0.set_ylabel("$v_\\phi/c_s$")
        #ax0.set_xlim(np.min(ztesti),np.max(ztesti))
        #ax0.set_ylim(np.min(M),np.max(M))
        #if taussd[1]-taussd[0] !=taussd[2]-taussd[1]:
        #    print('xscale'
        #    ax0.set_xscale('log')
        fig.canvas.draw()
        plt.show()

        plt.rcParams.update({'font.size': 25})
        fig = plt.figure(figure+4,figsize=[8,6])
        fig.clf()
        ax0 = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.18, right=0.9, top=0.9, bottom=0.2)
        data=-kpararpp.T/k0
        if kmax is not None:  
            cf0 = plt.pcolor(k/k0,vphi/cs,data,cmap=plt.cm.gist_earth_r,vmin =-kmax, vmax =kmax)
        else:
            cf0 = plt.pcolor(k/k0,vphi/cs,data,cmap=plt.cm.gist_earth_r)
        if kticks is not None:
            plt.colorbar(cf0, ticks=kticks)
        else:
            plt.colorbar(cf0)
        ax0.set_xlabel("$\\vert k_s\\vert /k_0$")
        ax0.set_ylabel("$v_\\phi/c_s$")
        #ax0.set_xlim(np.min(ztesti),np.max(ztesti))
        #ax0.set_ylim(np.min(M),np.max(M))
        #if taussd[1]-taussd[0] !=taussd[2]-taussd[1]:
        #    print('xscale'
        #    ax0.set_xscale('log')
        fig.canvas.draw()
        plt.show()

def dispe_kinSSD_vsm(m,vphiscs=1,Te=1.e3,Ti=[300.],Z=[1.],A=[1.],nesnc=0.1,I0=3e14,f=8.,wm=2*np.pi*14.25e9,Tr=70e-12,k0=2*np.pi/0.35e-6,nisne=None,isnonlocal=True,k=None,figure=1,gmax=None,kmax=None,gticks=None,kticks=None):
    #dispe_filam_kinRPP(Te=700.,Ti=[500,500.],Z=[1.,6.],A=[1.,12.],nisne=[1./7.,1./7.],gmax=3.e-2,kmax=0.1,gticks=[0,0.01,0.02,0.03,0.04,0.05],kticks=[-0.1,0,0.1])
    #dispe_filam_kinRPP(Te=1000.,Ti=[300],Z=[1.],A=[1.],gmax=8e-3,kmax=0.1,gticks=[0,0.002,0.004,0.006,0.008],kticks=[-0.1,0,0.1])
    c=3e8
    km=k0/(2*f)
    nc=k0**2*c**2*8.85e-12*9.1e-31/(1.6e-19)**2
    print('nc = ',nc, ' m^-3')
    if nisne is None:
        nisne = [1./np.float(np.array(Z))]
        print('nisne = ',nisne)
    dnsn = I0*1e4/(nc*c*Te*1.6e-19)
    print('dnsn = ',dnsn)
    n=np.sqrt(1-nesnc)
    w0=k0*c/n
    Tim, Am, Zm = np.mean(np.array(Ti)),np.mean(np.array(A)),np.mean(np.array(Z))
    cs = np.sqrt((Zm*Te+3*Tim)/(1836.*511000.*Am ))*c
    zeta= Zm*Te/511000./(1836*Am*cs**2/c**2)

    mm=0
    for i in range(len(Ti)):
        mm+=Z[i]**2*nisne[i]*Te/Ti[i]
    alphak = m/(1.+m)
    
    ne=nesnc*nc/1e6
    print([1.,A[0]*1836.],[-1.,Z[0]],[ne,ne/Z[0]],[Te,Ti[0]])
    r=taux_collisionnel(masse=[1.,A[0]*1836.],charge=[-1.,Z[0]],dens=[ne,ne*nisne[0]],temp=[Te,Ti[0]],vd=[0,0])
    print(r['lmfp'])
    lmfp=r['lmfp']*1e-2
    iak=0
    if len(Z)>1:
        for i in range(1,len(Z)):
            r=taux_collisionnel(masse=[1.,A[i]*1836.],charge=[-1.,Z[i]],dens=[ne,ne*nisne[i]],temp=[Te,Ti[i]],vd=[0,0])
            print('lmfp, i = ',i, '   : ',r['lmfp'])
            if lmfp  > r['lmfp']*1e-2:
                lmfp = r['lmfp']*1e-2 
                iak  = i

    print('lmfp = ',lmfp,' m')

    #Brillouin avant
    if k is None:
        k = np.linspace(0.03*k0*dnsn**0.5,3*k0*dnsn**0.5,100)
    x=lmfp*Z[i]**0.5*k
    Ak =( 0.5 +isnonlocal*Z[i]*(0.074/x**2 +0.88/x**(4./7.) +2.54/(1.+5.5*x**2)) )

    #g0= np.sqrt(np.pi) *(cs/c)**3*(np.exp(-Am*1836*9.1e-31*cs**2/(2*Tim*1.6e-19))/(2*Tim/511000.)**1.5 +Am*1836./Zm/(2*Te/511000.)**1.5)
    vi=np.sqrt(Tim/(Am*1836*511000.))*c
    g0= np.sqrt(np.pi/8) *(cs/vi)**3*(np.exp(-0.5*(cs/vi)**2 ) +np.sqrt(Zm/(Am*1836.)) *(Zm*Te/Tim)**-1.5)

    print('g_0 = ', g0)

    vphi=vphiscs*cs
    
    V, K=np.meshgrid(m,k)
   
    U = 0*V +0j*V
    for im in range(len(m)):
        print('m : ',im,' / ',len(m))
        for ik in range(len(k)): 
            #cinetique RPP+SSD
            betak = 0
            for l1 in range(-int(m[im]),int(m[im])+1):
                for l2 in range(-int(m[im]),int(m[im])+1):
                    vpt = (vphi*k[ik] + wm*(l1-l2))/np.sqrt(k[ik]**2 + wm**2/w0**2*k0**2*(l1-l2)**2)
                    alphak = alpha_kin(Te,Ti, Z, A, nisne, vpt, k2lde2=0, ne=1)
                    arg=np.pi*0.5*Tr*wm*(l1-l2)
                    betak +=jv(l1,m[im])*jv(l2,m[im])*np.sinc(arg/np.pi)*np.exp(1j*arg)*alphak
            B = - (4*km*k[ik]*c**2 )/w0**2*(1. /( nesnc*dnsn/n**2*betak*Ak[ik])) 
            exp = np.exp( B)
            if np.real(B)>1e20:
                print('-1/B = ', B)
                E=  ( (k[ik]/k0+1./f )**2  )
            if np.abs(B)<1e-10:
                E=  ( (k[ik]/k0-1./f)- (1+ B)*(k[ik]/k0+1./f)**2  ) /(- B)
            else:
                exp = np.exp(  B)
                #print(B,np.exp(B))
                E=  ( (k[ik]/k0-1./f) - exp*(k[ik]/k0+1./f ) ) /(1-exp)
            U[ik,im] = vphi/c/n+0.5*E
        
            
    gamma = -K*np.imag(U)
    kpara = K*np.real(U)
    
    if np.sum(np.abs(gamma))== 0 :
        print('No instability with ssd !')
    else:
        print('Max Gamma/k0 = ' , np.max(gamma)/k0)
    

    for im in range(len(m)):
        for ik in range(len(k)):
            if np.isnan(gamma[ik,im]) or np.isnan(kpara[ik,im]):
                gamma[ik,:im] = 0
                kpara[ik,:im] = 0

    if figure is not None:
       

        plt.rcParams.update({'font.size': 25})
        fig = plt.figure(figure+1,figsize=[8,6])
        fig.clf()
        ax0 = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.18, right=0.9, top=0.9, bottom=0.2)
        data=gamma.T/k0
        #data = np.log10(gamma)
        if gmax is not None:  
            cf0 = plt.pcolor(k/k0,m,data,cmap=plt.cm.gist_earth_r,vmin =0, vmax =gmax)
        else:
            cf0 = plt.pcolor(k/k0,m,data,cmap=plt.cm.gist_earth_r)
        if gticks is not None:
            plt.colorbar(cf0, ticks=gticks)
        else:
            plt.colorbar(cf0)
        ax0.set_xlabel("$\\vert k_s\\vert /k_0$")
        ax0.set_ylabel("$m$")
        #ax0.set_xlim(np.min(ztesti),np.max(ztesti))
        #ax0.set_ylim(np.min(M),np.max(M))
        #if taussd[1]-taussd[0] !=taussd[2]-taussd[1]:
        #    print('xscale'
        #    ax0.set_xscale('log')
        fig.canvas.draw()
        plt.show()

        plt.rcParams.update({'font.size': 25})
        fig = plt.figure(figure+2,figsize=[8,6])
        fig.clf()
        ax0 = fig.add_subplot(1,1,1)
        plt.subplots_adjust(left=0.18, right=0.9, top=0.9, bottom=0.2)
        data=-kpara.T/k0
        #data = np.log10(gamma)
        if kmax is not None:  
            cf0 = plt.pcolor(k/k0,m,data,cmap=plt.cm.gist_earth_r,vmin =-kmax, vmax =kmax)
        else:
            cf0 = plt.pcolor(k/k0,m,data,cmap=plt.cm.gist_earth_r)
        if kticks is not None:
            plt.colorbar(cf0, ticks=kticks)
        else:
            plt.colorbar(cf0)
        ax0.set_xlabel("$\\vert k_s\\vert /k_0$")
        ax0.set_ylabel("$v_\\phi/c_s$")
        #ax0.set_xlim(np.min(ztesti),np.max(ztesti))
        #ax0.set_ylim(np.min(M),np.max(M))
        #if taussd[1]-taussd[0] !=taussd[2]-taussd[1]:
        #    print('xscale'
        #    ax0.set_xscale('log')
        fig.canvas.draw()
        plt.show()
