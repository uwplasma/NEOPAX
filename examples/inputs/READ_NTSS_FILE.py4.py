#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
import numpy as np
from scipy.io import netcdf
import h5py as h5
import os
import math
import sys
from pathlib import Path
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline

one = 1.
zero = 0.
matplotlib.rcParams.update({'font.size': 10,'font.weight': 'bold'})

def main(file,name='',figures_folder='.', coils_curves=None, s_plot_ignore=0.2,savefig=True):

    filename = file
    if name=='': name=os.path.basename(filename)[5:-3]
    #config_name=filename.split('wout_')[1].split('.nc')[0]

    #define required r_N=sqrt(s)
    r_N=np.array([0.12247,0.25, 0.375 ,0.5,0.625, 0.75, 0.875])

    ftext1=open(filename,'r')
    #ftext=open('runspec.dat','w')
    ftext1.readline() # line with variable names


    r=np.zeros((51,))
    #Densities
    ne=np.zeros((51,))
    nD=np.zeros((51,))
    nT=np.zeros((51,))
    nHe=np.zeros((51,))
    #Temperatures
    Te=np.zeros((51,))
    Td=np.zeros((51,))
    Tt=np.zeros((51,))
    #Other quantities
    Er=np.zeros((51,))
    Zeff=np.zeros((51,))
    Pressure=np.zeros((51,))
    NBIPower_e=np.zeros((51,))
    NBIPower_I=np.zeros((51,))
    ECRHPower=np.zeros((51,))
    AlphaPower=np.zeros((51,))
    D11e=np.zeros((51,))
    D12e=np.zeros((51,))
    D21e=np.zeros((51,))
    D22e=np.zeros((51,))
    D11d=np.zeros((51,))
    D12d=np.zeros((51,))
    D21d=np.zeros((51,))
    D22d=np.zeros((51,))
    D11t=np.zeros((51,))
    D12t=np.zeros((51,))
    D21t=np.zeros((51,))
    D22t=np.zeros((51,))
    D31e=np.zeros((51,))
    D32e=np.zeros((51,))
    D31d=np.zeros((51,))
    D32d=np.zeros((51,))
    D31t=np.zeros((51,))
    D32t=np.zeros((51,))

    Dpsi=np.zeros((51,))
    Sigma=np.zeros((51,))
    DeAno=np.zeros((51,))
    XeAno=np.zeros((51,))
    XiAno=np.zeros((51,))

    beta=np.zeros((51,))

    nu_e=np.zeros((51,))
    nu_d=np.zeros((51,))
    nu_t=np.zeros((51,))
    roStarD=np.zeros((51,))
    roStarT=np.zeros((51,))

    FluxN=np.zeros((51,))
    FluxQe=np.zeros((51,))
    FluxQI=np.zeros((51,))
    FluxNeo=np.zeros((51,))
    FluxAno=np.zeros((51,))
    FluxQeNeo=np.zeros((51,))
    FluxQiNeo=np.zeros((51,))
    FluxQeAno=np.zeros((51,))
    FluxQiAno=np.zeros((51,))

    AmbiFlux=np.zeros((51,))
    J_bse=np.zeros((51,))
    J_bsi=np.zeros((51,))
    J_bs=np.zeros((51,))
    J_eccd=np.zeros((51,))
    J_nbcd=np.zeros((51,))
    J_cd=np.zeros((51,))
    J_ohm=np.zeros((51,))
    J_tor=np.zeros((51,))
    I_bs=np.zeros((51,))
    I_eccd=np.zeros((51,))
    I_nbcd=np.zeros((51,))
    I_bs=np.zeros((51,))
    I_cd=np.zeros((51,))
    I_tor=np.zeros((51,))
    I_ohm=np.zeros((51,))

    iotaCF=np.zeros((51,))          
    iotaBS=np.zeros((51,))
    iota=np.zeros((51,))
    Vr=np.zeros((51,))



    for i in range(51):
            
        line=ftext1.readline()
        line1=line.split()
        r[i]=float(line1[0])
        #Densities
        ne[i]=float(line1[1])
        nD[i]=float(line1[2])
        nT[i]=float(line1[3])
        nHe[i]=float(line1[4])
        #Temperatures
        Te[i]=float(line1[5])
        Td[i]=float(line1[6])
        Tt[i]=float(line1[7])
        #Other quantities
        Er[i]=float(line1[8])
        Zeff[i]=float(line1[9])
        Pressure[i]=float(line1[10])
        NBIPower_e[i]=float(line1[11])
        NBIPower_I[i]=float(line1[12])
        ECRHPower[i]=float(line1[13])
        AlphaPower[i]=float(line1[14])
        D11e[i]=float(line1[15])
        D12e[i]=float(line1[16])
        D21e[i]=float(line1[17])
        D22e[i]=float(line1[18])
        D11d[i]=float(line1[19])
        D12d[i]=float(line1[20])
        D21d[i]=float(line1[21])
        D22d[i]=float(line1[22])
        D11t[i]=float(line1[23])
        D12t[i]=float(line1[24])
        D21t[i]=float(line1[25])
        D22t[i]=float(line1[26])
        D31e[i]=float(line1[27])
        D32e[i]=float(line1[28])
        D31d[i]=float(line1[29])
        D32d[i]=float(line1[30])
        D31t[i]=float(line1[31])
        D32t[i]=float(line1[32])

        Dpsi[i]=float(line1[33])
        Sigma[i]=float(line1[34])
        DeAno[i]=float(line1[35])
        XeAno[i]=float(line1[36])
        XiAno[i]=float(line1[37])

        beta[i]=float(line1[38])

        nu_e[i]=float(line1[39])
        nu_d[i]=float(line1[40])
        nu_t[i]=float(line1[41])
        roStarD[i]=float(line1[42])
        roStarT[i]=float(line1[43])

        FluxN[i]=float(line1[44])
        FluxQe[i]=float(line1[45])
        FluxQI[i]=float(line1[46])
        FluxNeo[i]=float(line1[47])
        FluxAno[i]=float(line1[48])
        FluxQeNeo[i]=float(line1[49])
        FluxQiNeo[i]=float(line1[50])
        FluxQeAno[i]=float(line1[51])
        FluxQiAno[i]=float(line1[52])

        AmbiFlux[i]=float(line1[53])
        J_bse[i]=float(line1[54])
        J_bsi[i]=float(line1[55])
        J_bs[i]=float(line1[56])
        J_eccd[i]=float(line1[57])
        J_nbcd[i]=float(line1[58])
        J_cd[i]=float(line1[59])
        J_ohm[i]=float(line1[60])
        J_tor[i]=float(line1[61])
        I_bs[i]=float(line1[62])
        I_eccd[i]=float(line1[63])
        I_nbcd[i]=float(line1[64])
        I_cd[i]=float(line1[65])
        I_tor[i]=float(line1[66])
        I_ohm[i]=float(line1[67])

        iotaCF[i]=float(line1[68])      
        iotaBS[i]=float(line1[69])
        iota[i]=float(line1[70])
        Vr[i]=float(line1[71])

    ftext1.close()

    ntss=h5.File('NTSS_'+filename+'.h5','w')

    ntss['r']=r
    ntss['ne']=ne
    ntss['nD']=nD
    ntss['nHe']=nHe

    #Temperatures
    ntss['Te']=Te
    ntss['TD']=Td
    ntss['Tt']=Tt

    #Other quantities
    ntss['Er']=Er
    ntss['Zeff']=Zeff
    ntss['Pressure']=Pressure
    ntss['NBIPower_e']=NBIPower_e
    ntss['NBIPower_I']=NBIPower_I
    ntss['ECRHPower']=ECRHPower
    ntss['AlphaPower']=AlphaPower
    ntss['D11e']=D11e
    ntss['D12e']=D12e
    ntss['D21e']=D21e
    ntss['D22e']=D22e
    ntss['D11d']=D11d
    ntss['D12d']=D12d
    ntss['D21d']=D21d
    ntss['D22d']=D22d
    ntss['D11t']=D11t
    ntss['D12t']=D12t
    ntss['D21t']=D21t
    ntss['D22t']=D22t
    ntss['D31e']=D31e
    ntss['D32e']=D32e
    ntss['D31d']=D31d
    ntss['D32d']=D32d
    ntss['D31t']=D31t
    ntss['D32t']=D32t

    ntss['Dpsi']=Dpsi
    ntss['Sigma']=Sigma
    ntss['DeAno']=DeAno
    ntss['XeAno']=XeAno

    ntss['beta']=beta

    ntss['nu_e']=nu_e
    ntss['nu_d']=nu_d
    ntss['nu_t']=nu_t
    ntss['roStarD']=roStarD
    ntss['roStarT']=roStarT

    ntss['FluxN']=FluxN
    ntss['FluxQe']=FluxQe
    ntss['FluxQI']=FluxQI
    ntss['FluxNeo']=FluxNeo
    ntss['FluxAno']=FluxAno
    ntss['FluxQeNeo']=FluxQeNeo
    ntss['FluxQiNeo']=FluxQiNeo
    ntss['FluxQeAno']=FluxQeAno
    ntss['FluxQiAno']=FluxQiAno

    ntss['AmbiFlux']=AmbiFlux
    ntss['J_bse']=J_bse
    ntss['J_bsi']=J_bsi
    ntss['J_bs']=J_bs
    ntss['J_eccd']=J_eccd
    ntss['J_nbcd']=J_nbcd
    ntss['J_cd']=J_cd
    ntss['J_ohm']=J_ohm
    ntss['J_tor']=J_tor
    ntss['I_bs']=I_bs
    ntss['I_eccd']=I_eccd
    ntss['I_nbcd']=I_nbcd
    ntss['I_cd']=I_cd
    ntss['I_tor']=I_tor
    ntss['I_ohm']=I_ohm

    ntss['iotaCF']=iotaCF
    ntss['iotaBS']=iotaBS
    ntss['iota']=iota
    ntss['Vr']=Vr

    maxEr=np.max(Er)
    Er_change=[]
    Er_change_abs=[]
    idx = np.where(np.sign(Er[:-1]) != np.sign(Er[1:]))[0] + 1
    print(idx)
    for i in idx:
    #    if i-1 !=0:
        trans_abs=(r[i]-r[i-1])*0.5+r[i-1]
        trans=trans_abs/r[-1]
        Er_change.append(trans)
        Er_change_abs.append(trans_abs)
    Er_trans=np.array(Er_change)
    Er_trans_abs=np.array(Er_change_abs)        
    ntss['maxEr']=maxEr 
    ntss['Er_trans']=Er_trans
    ntss['Er_trans_abs']=Er_trans_abs


    ntss.close()
    fig1 = plt.figure()
    #fig1.set_size_inches(14,7)
    #fig1.patch.set_facecolor('white')
    plt.plot(r,Er,'r')
    #plt.title(titles[i])#+'\n1-based index='+str(iradius+1))
    plt.xlabel('$r [m]$')
    plt.ylabel('$ E_r [kV/m] $')
    #plt.xlim([0,2*np.pi])
    #    plt.ylim([0,2*np.pi])

    #plt.tight_layout()
        #plt.figtext(0.5,0.99,os.path.abspath(filename),ha='center',va='top',fontsize=6)

    plt.savefig('Er'+filename+'.pdf', bbox_inches = 'tight', pad_inches = 0)

    fig1 = plt.figure()
    plt.plot(r,FluxQI*Vr/(337.919-35.451),'b',label='$Q_i$')
    plt.plot(r,FluxQe*Vr/(337.919-35.451),'r',label='$Q_e$')
    plt.plot(r,(FluxQe+FluxQI)*Vr/(337.919-35.451),'k',label='$Q_i+Q_e$')
    plt.xlabel('$r [m]$')
    plt.ylabel('$ Q_a/P_{\\alpha} $')
    plt.legend()
    plt.savefig('Heat'+filename+'.pdf', bbox_inches = 'tight', pad_inches = 0)

    fig1 = plt.figure()
    plt.plot(r,FluxQI*Vr,'b',label='$Q_I$')
    plt.plot(r,FluxQe*Vr,'r',label='$Q_e$')
    plt.plot(r,(FluxQe+FluxQI)*Vr,'k',label='$Q_T$')
    plt.xlabel('$r [m]$')
    plt.ylabel('$ Heat flux [MW] $')
    plt.legend()
    plt.savefig('Heat_Heat'+filename+'.pdf', bbox_inches = 'tight', pad_inches = 0)


    fig1 = plt.figure()
    plt.plot(r,AlphaPower,'r')
    plt.xlabel('$r [m]$')
    plt.ylabel('$ Alpha power [MW] $')

    plt.savefig('Alpha'+filename+'.pdf', bbox_inches = 'tight', pad_inches = 0)

    fig1 = plt.figure()
    plt.plot(r,FluxQI/AlphaPower,'b',label='$Q_I$')
    plt.plot(r,FluxQe/AlphaPower,'r',label='$Q_e$')
    plt.plot(r,(FluxQe+FluxQI)/AlphaPower,'k',label='$Q_T$')
    plt.yscale('log')
    plt.xlabel('$r [m]$')
    plt.ylabel('$ Heat flux/P_{\\alpha} $')
    plt.legend()
    plt.savefig('Heat_relative'+filename+'.pdf', bbox_inches = 'tight', pad_inches = 0)
    
    
    
    fig1 = plt.figure()
    plt.plot(r,ne,'k',label='$n_e [10^{20} m^{-3}]$')
    plt.plot(r,nD,'r',label='$n_D=n_T  [10^{20} m^{-3}]$')
    plt.plot(r,nHe,'b',label='$n_{He} [10^{20} m^{-3}]$')
    plt.legend()
    plt.xlabel('$r [m]$')
    plt.ylabel('$ n [ 10^{20} m^{-3}] $')
    plt.savefig('Density'+filename+'.pdf', bbox_inches = 'tight', pad_inches = 0)


    fig1 = plt.figure()
    plt.plot(r,Te,'k',label='$T_e [keV]$')
    plt.plot(r,Td,'r',label='$T_D [keV]$')
    plt.plot(r,Tt,'b',label='$T_T [keV]$')
    plt.legend()
    plt.xlabel('$r [m]$')
    plt.ylabel('$ T [keV] $')
    plt.savefig('Temperature'+filename+'.pdf', bbox_inches = 'tight', pad_inches = 0)

    fig1 = plt.figure()
    plt.plot(r,Zeff,'k')
    plt.xlabel('$r [m]$')
    plt.ylabel('$Z_{eff}$')

    plt.savefig('Zeff'+filename+'.pdf', bbox_inches = 'tight', pad_inches = 0)
    print('Max Er', maxEr)
    print('Trans abs Er', Er_trans_abs)
    print('Max Er', Er_trans)
    print('NTSS file read and h5 ceated')

    fig1 = plt.figure()
    plt.plot(r,FluxQiNeo*Vr/(337.919-35.451),'b',label='$Q_i^{NEO}$')
    plt.plot(r,FluxQeNeo*Vr/(337.919-35.451),'r',label='$Q_e^{NEO}$')
    plt.plot(r,(FluxQeNeo+FluxQiNeo)*Vr/(337.919-35.451),'k',label='$Q_i^{NEO}+Q_e^{NEO}$')
    plt.xlabel('$r [m]$')
    plt.ylabel('$ Q_a^{NEO}/P_{\\alpha}$')
    plt.legend()
    plt.savefig('Heat_HeatNeo'+filename+'.pdf', bbox_inches = 'tight', pad_inches = 0)
    
    fig1 = plt.figure()
    plt.plot(r,FluxQiAno*Vr/(337.919-35.451),'b',label='$Q_I$')
    plt.plot(r,FluxQeAno*Vr/(337.919-35.451),'r',label='$Q_e$')
    plt.plot(r,(FluxQeAno+FluxQiAno)*Vr/(337.919-35.451),'k',label='$Q_T$')
    plt.xlabel('$r [m]$')
    plt.ylabel('$ Heat flux [MW] $')
    plt.legend()
    plt.savefig('Heat_HeatAno'+filename+'.pdf', bbox_inches = 'tight', pad_inches = 0)


    fig1 = plt.figure()
    plt.plot(r,AmbiFlux,'b',label='$Gamma$')
    plt.xlabel('$r [m]$')
    plt.ylabel('$ Flux  [MW] $')
    plt.legend()
    plt.savefig('FluxNeo'+filename+'.pdf', bbox_inches = 'tight', pad_inches = 0)


    rho=r/r[-1]
    S_a=AlphaPower
    Tau_He=np.sum(rho*nHe)/np.sum(rho*S_a)
    Hefinal=S_a*Tau_He/nHe

    He_file=h5.File('Helium.h5','w')
    He_file['r']=rho    
    He_file['S_He']=S_a
    He_file['Tau_He']=Tau_He
    He_file['Hefinal']=Hefinal 
    He_file.close()    

    fig1 = plt.figure()
    plt.plot(r[:-2],Hefinal[:-2],'b',label='$Hefinal$')
    plt.xlabel('$r [m]$')
    plt.ylabel('$ S_{\\alpha}\\tau_{He}/n_{He}$')
    plt.legend()
    plt.savefig('HeFinal'+filename+'.pdf', bbox_inches = 'tight', pad_inches = 0)
    Palhavol=(337.919-35.451)
    print(Palhavol)

if __name__ == "__main__":
    # Create results folders if not present
    try:
        Path(sys.argv[2]).mkdir(parents=True, exist_ok=True)
        figures_results_path = str(Path(sys.argv[2]).resolve())
        main(sys.argv[1], sys.argv[2])
    except:
        main(sys.argv[1])