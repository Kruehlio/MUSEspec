# -*- coding: utf-8 -*-

""" 
Formulas to derive physical quantities from line fluxes

"""

import numpy as np

def calcebv(ha, hb):
    kha, khb, khg, khd = 2.446, 3.560, 4.019, 4.253
    Cha, Chb, Chg, Chd = 1, 0.348, 0.162, 0.089
    ebv = np.log10((Cha/Chb)/(ha/hb)) / (0.4*(kha-khb))
    return ebv

def mcebv(ha, hb, n=1E4):
    haa = np.random.normal(np.ones(n) * ha[0], ha[1])
    hba = np.random.normal(np.ones(n) * hb[0], hb[1])
    ebv = calcebv(haa, hba)
    return np.nanmedian(ebv), np.nanstd(ebv)  


def mcSFR(ha, hb, s3d, n=1E5):
    haa = np.random.normal(np.ones(n) * ha[0], ha[1])
    hba = np.random.normal(np.ones(n) * hb[0], hb[1])
    ebv = calcebv(haa, hba)
    hacor = haa * s3d.ebvCor('ha', ebv=ebv)
    sfr = 4 * np.pi * s3d.LDMP**2 * hacor * 1E-17 * 4.8E-42
    return np.nanmedian(sfr), np.nanstd(sfr)  


def calcohD16(siia, siib, ha, nii):
    s2 = np.log10(nii/(siia+siib)) + 0.264 * np.log10(nii/ha)
    return 8.77 + s2 + 0.45 * (s2 + 0.3)**5

def mcohD16(siia, siib, ha, niib, n = 1E4):
    siiaa = np.random.normal(np.ones(n) * siia[0], siia[1])
    siiba = np.random.normal(np.ones(n) * siib[0], siib[1])
    niiba = np.random.normal(np.ones(n) * niib[0], niib[1])
    haa = np.random.normal(np.ones(n) * ha[0], ha[1])
    oh = calcohD16(siiaa, siiba, haa, niiba)
    return np.nanmedian(oh), np.nanstd(oh)
    
def calcohPP04(oiii, hb, nii, ha):
    o3n2 = np.log10((oiii/hb)/(nii/ha))
    n2 = np.log10(nii/ha)
    return 8.73 - 0.32 * o3n2, 9.37 + 2.03*n2 + 1.26*n2**2 + 0.32*n2**3

def mcohPP04(oiii, hb, nii, ha, n = 1E4):
    oiiia = np.random.normal(np.ones(n) * oiii[0], oiii[1])
    hba = np.random.normal(np.ones(n) * hb[0], hb[1])
    niia = np.random.normal(np.ones(n) * nii[0], nii[1])
    haa = np.random.normal(np.ones(n) * ha[0], ha[1])
    oho3n2, ohn2 = calcohPP04(oiiia, hba, niia, haa)
    return (np.nanmedian(oho3n2), np.nanstd(oho3n2), 
            np.nanmedian(ohn2), np.nanstd(ohn2))
    
def calcTSIII(siii, siiiau):
    a, b, c, d = 10719, 0.09519, 1.03510, 6.5E-3
    a1, a2, a3 = 1.00075, 1.09519, 3.21668
    b1, b2, b3 = 13.3016, 24396.2, 57160.4       
    t, kappa = 1E4, 20
    
    for n in range(40):
        t = a * (-np.log10((siiiau/siii)/(1 + d*(100./t**0.5))) - b)**(-c) 
    
    a = (a1 + a2/kappa + a3/kappa**2)
    b = (b1 + b2/kappa + b3/kappa**2)
    tkin = a * t - b
    return t, tkin
    
    
def mcTSIII(siii, siiiau, n=1E4):
    siiia = np.random.normal(np.ones(n) * siii[0], siii[1])
    siiiaua = np.random.normal(np.ones(n) * siiiau[0], siiiau[1])
    Ta = calcTSIII(siiia, siiiaua)
    return np.nanmedian(Ta[0]), np.nanstd(Ta[0])
    

def calcOHOIII(oiii, oiia, oiib, hb, toiii, toii):
    x = 10**(-4)*100*toiii**(-0.5)
    logoih = np.log10((oiia+oiib)/hb) + 6.901 + 2.487 / toii\
        - 0.483 * np.log10(toii) - 0.013*toii + np.log10(1 - 3.48*x)       
    logoiih = np.log10(1.33*oiii/hb) + 6.200 + 1.251 / toiii \
        - 0.55 * np.log10(toiii) - 0.014 * toiii 
    return np.log10((10**(logoih-12)+10**(logoiih-12)))+12


def mcOHOIII(oiii, oiia, oiib, hb, toiii, toiiie, n=1E4):
    oiiia = np.random.normal(np.ones(n) * oiii[0], oiii[1])
    oiiaa = np.random.normal(np.ones(n) * oiia[0], oiia[1])
    oiiba = np.random.normal(np.ones(n) * oiib[0], oiib[1])
    hba = np.random.normal(np.ones(n) * hb[0], hb[1])
    toiiia = np.random.normal(np.ones(n) * toiii, toiiie)
    toiia = (-0.744 + toiii/1E4*(2.338 - 0.610*toiii/1E4)) * 1E4
    ohtoiii = calcOHOIII(oiiia, oiiaa, oiiba, hba, toiiia/1E4, toiia/1E4)
    return np.nanmedian(ohtoiii), np.nanstd(ohtoiii)
    
    
def calcSHSIII(siiib, siia, siib, hb, tsiii, toii):
    x = 10**(-4)*100*tsiii**(-0.5)
    logsih = np.log10((siia+siib)/hb) + 5.439 + 0.929 / toii\
        - 0.28 * np.log10(toii) - 0.018*toii + np.log10(1 + 1.39*x)       
    logsiih = np.log10(siiib/hb) + 6.690 + 1.678 / tsiii \
        - 0.47 * np.log10(tsiii) - 0.010 * tsiii 
    logsh = np.log10((10**(logsiih-12)+10**(logsih-12)))+12
    return logsh, logsih, logsiih   

    
def mcSHSIII(siiib, siia, siib, hb, tsiii, tsiiie, toiii, n=1E4):
    siiiba = np.random.normal(np.ones(n) * siiib[0], siiib[1])
    siiaa = np.random.normal(np.ones(n) * siia[0], siia[1])
    siiba = np.random.normal(np.ones(n) * siib[0], siib[1])
    hba = np.random.normal(np.ones(n) * hb[0], hb[1])
    tsiiia = np.random.normal(np.ones(n) * tsiii, tsiiie)
    toii = (-0.744 + toiii/1E4*(2.338 - 0.610*toiii/1E4)) * 1E4
    shtsiii = calcSHSIII(siiiba, siiaa, siiba, hba, tsiiia/1E4, toii/1E4)
    return np.nanmedian(shtsiii), np.nanstd(shtsiii)
    
def calcDens(siia, siib):
    nemap = 4.705 - 1.9875*siia/siib
    return nemap

def mcDens(siia, siib, n=1E4):
    siiaa = np.random.normal(np.ones(n) * siia[0], siia[1])
    siiba = np.random.normal(np.ones(n) * siib[0], siib[1])  
    dens = calcDens(siiaa, siiba)
    return np.nanmedian(dens), np.nanstd(dens)
   