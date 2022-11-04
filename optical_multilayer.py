#!/usr/bin/env python

# Authors: David C. Heson, Jack Liu, Dr. Bill Robertson, June 2022.

# Program to simulate reflection/transmission coefficients across custom multilayers.

import numpy as np
import matplotlib.pyplot as plt
import math
import time
import sys
import copy
import os 
import csv

### Optical Multilayer Functions:

def multilayer(n_list, dist, mode = 'rp', step = 1000, option_aw = 'a', e_opt = 'n',
               a1 = 0, a2 = np.pi/2, w_t = 650e-9, w1 = 400e-9, w2 = 750e-9,
               ang = np.pi/3, n_wv_dic = {}, nw_list = [], limit = 1000e-9):
       
### Parent function, asks for the refraction indeces of all passing mediums, and 
### the thickness of the layers the light passes through. Preferably to pass n_list as a
### 1-dimensional np array with complex dtype, if possible. Results will be 
### output as radians/meters vs R/T, in the columns 0 and 1 of the matrix that 
### the function returns. a1 and a2 give the starting and ending angular positions.
### The w1 and w2 arguments give the wavelength range, in meters. w_t is the constant 
### wavelength if the code is ran in angular mode. 'dist' is the list of layer thicknesses. 
### option_aw specifies whether the code iterates over wavelengths or angles, the default
### 'a' iterating over angles, and 'w' iterating over wavelengths.
### e_opt being set to 'y' would run the code to output the electric field profile over the
### multilayers with respect to the depth from the top, at a fixed wavelength w_t and 
### fixed angle ang. 
### 
### Advanced:
### If n_wv_dic is specified as ANY non-empty dictionary, the function will automatically
### expect n_wv_dic to contain a series of functions that return the n^2 for the specific
### material, the materials being specified by nw_list which will contain the keys for the
### functions in n_wv_dic, ordered with respect to the layers in dist. Currently, this is
### optimized SPECIFICALLY for wavelength-dependent n's. If you wish to use the
### dictionary feature for n's, n_list can be anything (an empty list [] is a good
### option). Also, make sure that the n's returned are all complex, even if it is +0j


    if (len(n_list) != len(dist) + 2) and (n_wv_dic == 0):
        print("Error, number of layers infered from depths and indexes given does not match.")
        sys.exit()
        
    destination = np.zeros(shape = (step, 2), dtype = complex)
    nmbr_layers = np.shape(n_list)[0]
    im = complex(0, 1)

    #################################################################################
    
    def multilayer_matrix(lmbd, theta): 
        
        ### Sub function, it does the actual calculation for the Fresnel coefficients for
        ### fixed angle and wavelength values. It is called by the conditional statements 
        ### below it. 
        ### Follows the process outlined in DOI:10.1364/AO.29001952
        
        n_0 = n_list[0]
        
        C = [
            [1, 0], 
            [0, 1]
        ]
        
        if mode == 'tp' or 'ts':
            t_list = []
        
        for s in range(0, nmbr_layers - 1):

            if s == 0:
                width = 0
            else:
                width = dist[s-1]
            
            n1 = n_list[s]
            n2 = n_list[s + 1]
            
            cost2 = (1 - ((n_0/n2) ** 2) * (np.sin(theta) ** 2)) ** (0.5)
            cost1 = (1 - ((n_0/n1) ** 2) * (np.sin(theta) ** 2)) ** (0.5)
            
            if mode == 'rp' or mode == 'tp':
                r = (n1 * cost2 - n2 * cost1) / (n1 * cost2 + n2 * cost1)
                           
            elif mode == 'rs' or mode == 'ts':
                r = (n1 * cost1 - n2 * cost2) / (n1 * cost1 + n2 * cost2)
                       
            else:
                print("Error, invalid polarization mode argument.")
                sys.exit()
                
            if mode == 'tp':
                t_list.append((2 * n1 * cost1) / (n1 * cost2 + n2 * cost1))
                
            elif mode == 'ts':
                t_list.append((2 * n1 * cost1) / (n1 * cost1 + n2 * cost2))
                
            delta = complex(n1 * cost1 * 2 * np.pi * width / lmbd, 0)
            
            factor = [
                [complex(np.exp(-im * delta)), complex(r * np.exp(-im * delta))],
                [complex(r * np.exp(im * delta)), complex(np.exp(im * delta))]
            ]
            
            C = np.dot(C, factor)
            
        a = C[0][0]
        c = C[1][0]
                
        r = c/a
    
        if mode == 'rp' or mode == 'rs':
            return abs(r) ** 2
                    
        elif mode == 'tp':
            return (np.real((cost2 * n2.conjugate()) / (n_0.conjugate() * np.cos(theta)))
                    * (abs((np.prod(t_list) / a)) ** 2))
        
        else:
            return (np.real((cost2 * n2) / (n_0 * np.cos(theta)))
                    * (abs((np.prod(t_list) / a)) ** 2))
    
    ###################################################################################
    
    def multilayer_aux(lmbd, th, d, nlayer):
              
        N = len(nlayer)
        wl = lmbd
        
        dtotal = np.real(sum(d))
        wki = 2 * np.pi / wl
        
        sinth = []
        costh = []
        wvi = []
        
        for m in range(0, len(nlayer)):
            sinth.append(np.sin(th) * nlayer[0] / nlayer[m])
            costh.append((1 - (nlayer[0] * np.sin(th)/nlayer[m]) ** 2) ** 0.5)
            wvi.append(wki * nlayer[m])
        
        rp = []
        tp = []
        rs = []
        ts = []
            
        for m in range(0, len(nlayer) - 1):
            
            cost1 = costh[m]
            cost2 = costh[m+1]
            n1 = nlayer[m]
            n2 = nlayer[m+1]
            
            rp_t = (n2 * cost1 - n1 * cost2) / (n1 * cost2 + n2 * cost1)
            tp_t = (2 * n1 * cost1) / (n1 * cost2 + n2 * cost1)
            rs_t = (n1 * cost1 - n2 * cost2) / (n1 * cost1 + n2 * cost2)
            ts_t = (2 * n1 * cost1) / (n1 * cost1 + n2 * cost2)
            
            rp.append(rp_t)
            tp.append(tp_t)
            rs.append(rs_t)
            ts.append(ts_t)
            
        r = []
        t = []
        rss = []
        tss = []
        
        for i in range(0, len(nlayer) - 1):
            r.append(rp[i])
            t.append(tp[i])
            rss.append(rs[i])
            tss.append(ts[i])
        
        rr=np.copy(r)
        tt=np.copy(t)
        rrss=np.copy(rss)
        ttss=np.copy(tss)
        
        for m in range(len(nlayer) - 3, -1, -1):
            decay = 2 * im * d[m+1] * costh[m+1] * wvi[m+1]
            decayr = np.exp(decay)
            decayt = np.exp(im * d[m+1] * costh[m+1] * wvi[m+1])
            
            topr = r[m] + rr[m+1] * decayr
            topt = t[m] * tt[m+1] * decayt
            bot = 1 + r[m] * rr[m+1] * decayr
        
            toprs = rss[m] + rrss[m+1] * decayr
            topts = tss[m] * ttss[m+1] * decayt
            bots = 1 + rss[m] * rrss[m+1] * decayr
        
            rr[m] = topr / bot
            tt[m] = topt / bot
            rrss[m]= toprs / bots
            ttss[m]= topts / bots
        
        
        efldi = np.zeros(shape = (len(nlayer), 1), dtype=complex)
        efldr = np.zeros(shape = (len(nlayer), 1), dtype=complex)
        efldis = np.zeros(shape = (len(nlayer), 1), dtype=complex)
        efldrs = np.zeros(shape = (len(nlayer), 1), dtype=complex)
        
        efldi[0] = 1.
        efldr[0] = rr[0] * efldi[0]
        
        efldi[len(nlayer) - 1] = tt[0] * efldi[0]
        efldr[len(nlayer) - 1] = 0
        
        efldis[0] = 1.
        efldrs[0] = rrss[0] * efldis[0]
        
        efldis[len(nlayer) - 1] = ttss[0] * efldis[0]
        efldrs[len(nlayer) - 1] = 0
        
        for m in range(0, len(nlayer) - 2):
            decayr = np.exp(2 * im * d[m+1] * costh[m+1] * wvi[m+1])
            decayt2 = np.exp(im * d[m] * costh[m] * wvi[m])
            
            efldi[m+1] = ((t[m]) / (1 + r[m] * rr[m+1] * decayr)) * (efldi[m] * decayt2)
            efldr[m+1] = ((t[m] * rr[m+1] * decayr) / (1 + r[m] * rr[m+1] * decayr)) * efldi[m] * decayt2
        
            efldis[m+1] = ((tss[m]) / (1 + rss[m] * rrss[m+1] * decayr)) * (efldis[m] * decayt2)
            efldrs[m+1] = ((tss[m] * rrss[m+1] * decayr) / (1 + rss[m] * rrss[m+1] * decayr)) * efldis[m] * decayt2
            
        bound = np.zeros(shape = (len(nlayer)+1 , 1))
        bound[0] = 0
        bound[1]=0
        bound[len(nlayer)] = dtotal + 50000
        
        for m in range (1, len(nlayer) - 1):
            bound[m+1] = bound[m] + np.real(d[m])
            
        stepz = (dtotal + limit)/step
        
        j = 1
        
        etotals = np.zeros(shape = (step, 2))
        etotalss = np.zeros(shape = (step, 2))
    
        for m in range (0, step):
            
            z = (m) * stepz
            etotals[m, 0] = z
            etotalss[m, 0] = z
            
            if z >= bound[j+1]:
                j += 1
                
        
            eti = (efldi[j] * np.exp(im * wvi[j] * costh[j] * (z - bound[j])))
            etr = (efldr[j] * np.exp(-1 * im * wvi[j] * costh[j] * (z - bound[j])))
            etis = (efldis[j] * np.exp(im * wvi[j] * costh[j] * (z - bound[j])))
            etrs = (efldrs[j] * np.exp(-1 * im * wvi[j] * costh[j] * (z - bound[j])))
        
            
            etotals[m,1] = abs((eti + etr)**2)
            etotalss[m,1] = abs((etis + etrs)**2)
            
        if (mode == 'rp') or (mode == 'tp'):
            return etotals
            
        elif (mode == 'rs') or (mode == 'ts'):
            return etotalss
        
        else:
            print("Error, invalid polarization mode argument.")
            sys.exit()
        
    ##################################################################################
    
    ### Conditional Statements:
    ### Here is where the multilayer function is called
        
    if (e_opt == 'y'):
        
        empty = [0]
        dist = np.hstack([empty, dist])
        dist = np.hstack([dist, empty])
        
        destination = multilayer_aux(w_t, ang, dist, n_list)
        
    elif option_aw == 'a':
    
        increment = (a2 - a1) / (step)
        
        for i in range(0, step):
            
            t = increment * i + a1
            
            destination[i, 0] = t
            destination[i, 1] = multilayer_matrix(w_t, t)
        
    elif option_aw == 'w':
        
        increment = ((w2 - w1) / step)
        
        if len(n_wv_dic) == 0: ### checks if dictionary is empty, if it is it runs the 'if' block
            
            for i in range(0, step):
                
                wv = w1 + i * increment
                destination[i, 0] = wv
                destination[i, 1] = multilayer_matrix(wv, ang)
                
        elif len(dist) != len(nw_list) - 2:
            print("Error, material list (nw_list) does not match in length with the depth list.")
            sys.exit()
            
        else: ### ran if the dictionary is NOT empty
            
            for i in range(0, step):
                
                n_list = np.zeros(shape = (len(nw_list), 1), dtype = complex)
                wv = w1 + i * increment
                cntr = 0 
                
                for p in nw_list:
                    f = n_wv_dic[p]
                    n_list[cntr, 0] = f(wv)
                    cntr += 1

                destination[i, 0] = wv
                destination[i, 1] = multilayer_matrix(wv, ang)
                
    else:
        print("Error, invalid angle/wavelength mode argument.")
        sys.exit()
        
    return destination
 
###########################################################################################

def bloch_wave(n_list, d, mode = 'rp', step = 1000, option_aw = 'a', a1 = 0,
                 a2 = np.pi/2, w_t = 623.8e-9, w1 = 400e-9, w2 = 750e-9, 
                 ang = np.pi/3, roof = 0.98, minimal = 0.6, perc_trav = 0.01, verb = 0):

    ### Function to detect if/where Bloch surface waves occur within a bandgap, given a
    ### variation of angles with a fixed wavelength (option_aw = 'a', a1, a2, w_t), or 
    ### vice-versa (option_w = 'w', w1, w2, ang). sens_d determines how many steps through
    ### the given interval have to be above the roof in reflectivity for a bandgap to be 
    ### detected, and minimal gives the minimal reflectivity drop to equal the drop with a
    ### Bloch surface wave present. The function returns where the Bloch surface waves
    ### occurs as the wavelength/angle, the rough coordinates defining the gap, and the
    ### minimum in reflectivity which is related to the Bloch Surface Wave. 

    diffs = []
    diffsw = []
    sens_d = int(step * perc_trav)
    sim = multilayer(n_list, d, mode, step = step, option_aw = option_aw, a1 = a1, a2 = a2,
                     w_t = w_t, w1 = w1, w2 = w2, ang = ang)
    wv = sim[:,0]
    rat = sim[:,1]
        
    c = 0
    st = -1
    pos = []
        
    for i in range(0, step):
        if rat[i] >= roof:
            c += 1
            if c == sens_d:
                st = i - sens_d + 1
                pos.append(st)
        else:
            if st != -1:
                pos.append(i)
                st = -1
            c = 0          
                
    if len(pos) == 0:
        if verb != 0:
            print("No bandgap found within the given region.")
        return False, False, False
        
    if len(pos) % 2 == 1:
        pos.append(step-1)
    valid = []
    surface_pos = []
    minimum = []
        
    for i in range(1, len(pos) - 1, 2):
        x1 = pos[i]
        x2 = pos[i+1]
        low = np.amin(rat[x1:x2])
        if (low < minimal) and ((x2 - x1) < (step * 0.002)):
            if verb != 0:
                print("Bloch surface wave between points " + str(x1) + " and " + str(x2) +
                        " with a mininum of " + str(low) + " found.")
            valid.append(pos[i])
            valid.append(pos[i+1])
            minimum.append(low)
            low = np.where(rat[x1:x2] == low)
            low = int(low[0]) + pos[i]
            surface_pos.append(low)
        else:
            continue
            
    if (len(surface_pos) == 0):
        if verb != 0:
            print("No bandgap with a Bloch surface wave found within the given region.")
        return False, False, False
    else:
        return surface_pos, valid, minimum

##########################################################################################

def swt_calc(n_list, d_list, pol, steps, change = 'a', a_i = 0, a_f = np.pi/2, 
             w_c = 623.8e-9, w_i = 400e-9, w_f = 750e-9, a_c = np.pi/3, verb = 0):
    
    ### Function to calculate the SWT for a specified multilayer. 
    ### n_list is the list of refractive indexes for the multilayer, d_list is the list of
    ### widths of the layers within the multilayer, pol is the polarization argument for the
    ### light that passes through the multilayer, change represents over what variable the
    ### code iterates (<a>ngle or <w>avelength). a_i and a_f are initial/final angles, and
    ### w_i and w_f are initial and final wavelengths, and w_c and a_c are the constant 
    ### wavelength and angles used for the opposite iterations. 
    ### if verb is set to 1, the code will print out the found SWTs, and if it cannot detect
    ### a Bloch Surface Wave for the given layer. 
    
    d_indlist = copy.copy(d_list)
    d_indlist[-1] = d_indlist[-1] + 10e-9
    
    x, width, low = bloch_wave(n_list, d_list, mode = pol, step = steps, option_aw = change,
                              a1 = a_i, a2 = a_f, w_t = w_c, w1 = w_i, w2 = w_f, ang = a_c,
                              roof = 0.9, minimal = 0.4)
    x_ind, width_ind, low_ind = bloch_wave(n_list, d_indlist, mode = pol, step = steps, 
                                    option_aw = change, a1 = a_i, a2 = a_f, w_t = w_c, 
                                    w1 = w_i, w2 = w_f, ang = a_c, roof = 0.9, minimal = 0.4)
    
    if width == False:
        print("Error: no Bloch Surface Wave detected in the given setup.")
        return False
    
    elif x_ind == False:
        
        print("Error: a Bloch Surface Wave was detected in the original setup, but the adjusted setup did not exhibit one.")
        print("This error might be fixable by increasing the resolution to 30000 or above.")
        return False
    
    elif change == 'a':
        
        diff = abs(x[0] - x_ind[0])
        diff = (diff * (a_f - a_i) / steps) * 180/np.pi
        swt = float(diff)
        
        if verb == 1:
            print("The degrees per SWT for this design is:", swt)
        
    elif change == 'w':
        
        diff = abs(x[0] - x_ind[0])
        diff = (diff * (w_f - w_i) / steps) * 10e9
        swt = float(diff)
        
        if verb == 1:
            print("The nanometers per SWT for this design is:", swt)
        
    else:
        print("Error, invalid polarization mode argument.")
        sys.exit()
    
    return swt

##########################################################################################

def riu_calc(n_list, d_list, pol, steps, change = 'a', a_i = 0, a_f = np.pi/2, 
             w_c = 623.8e-9, w_i = 400e-9, w_f = 750e-9, a_c = np.pi/3, verb = 0):
    
    ### Function to calculate the RIU for a specified multilayer. 
    ### n_list is the list of refractive indexes for the multilayer, d_list is the list of
    ### widths of the layers within the multilayer, pol is the polarization argument for the
    ### light that passes through the multilayer, change represents over what variable the
    ### code iterates (<a>ngle or <w>avelength). a_i and a_f are initial/final angles, and
    ### w_i and w_f are initial and final wavelengths, and w_c and a_c are the constant 
    ### wavelength and angles used for the opposite iterations. 
    ### if verb is set to 1, the code will print out the found RIUs, and if it cannot detect
    ### a Bloch Surface Wave for the given layer. 
    
    n_indlist = copy.copy(n_list)
    n_indlist[-1] = n_indlist[-1] + complex(0.01, 0)
    
    x, width, low = bloch_wave(n_list, d_list, mode = pol, step = steps, option_aw = change,
                              a1 = a_i, a2 = a_f, w_t = w_c, w1 = w_i, w2 = w_f, ang = a_c,
                              roof = 0.9, minimal = 0.4)
    x_ind, width_ind, low_ind = bloch_wave(n_indlist, d_list, mode = pol, step = steps, 
                                    option_aw = change, a1 = a_i, a2 = a_f, w_t = w_c, 
                                    w1 = w_i, w2 = w_f, ang = a_c, roof = 0.9, minimal = 0.4)
    
    if width == False:
        print("Error: no Bloch Surface Wave detected in the given setup.")
        return False
    
    elif x_ind == False:
        
        print("Error: a Bloch Surface Wave was detected in the original setup, but the adjusted setup did not exhibit one.")
        print("This error might be fixable by increasing the resolution to 30000 or above.")
        return False
    
    elif change == 'a':
        
        diff = abs(x[0] - x_ind[0])
        diff = (diff * (a_f - a_i) / steps) * 180/np.pi
        riu = float(diff / 0.01)
        
        if verb == 1:
            print("The degrees per RIU for this design is:", riu)
        
    elif change == 'w':
        
        diff = abs(x[0] - x_ind[0])
        diff = (diff * (w_f - w_i) / steps) * 10e9
        riu = float(diff / 0.01)
        
        if verb == 1:
            print("The nanometers per SWT for this design is:", riu)
        
    else:
        print("Error, invalid polarization mode argument.")
        sys.exit()
    
    return riu

##########################################################################################

def graph(coord_list, label_list, size = (12, 6), efield = 0, d_set = None):
    
    ### Basic function to graph results, taking in a list that has numpy arrays of 
    ### the data to graph, the first column of each corresponding to x coordinates, and 
    ### the second column of which corresponding to y coordinates. The label list gives 
    ### what labels each set of coordinates should have. 
    ### Has a capability to be used specifically for electrical field simulations,
    ### graphing vertical lines that 
    
    fig, ax = plt.subplots(figsize=(size))
    for i in range(0, len(label_list)):
        curr_list = coord_list[i]
        if efield == 1:
            a = 0
            d_curr = d_set[i]
            for p in d_curr:
                a = p + a
                plt.axvline(x = a, color = "black", linestyle = '--')
        plt.plot(curr_list[:,0], curr_list[:,1], label = str(label_list[i]))
            
    plt.legend()
    plt.grid()
    plt.show()
    
###########################################################################################

def multilayer_explore(n_list, pol, steps, change = 'a', 
              a_i = 0, a_f = np.pi/2, w_c = 623.8e-9, w_i = 400e-9, w_f = 750e-9, 
              a_c = np.pi/3, def_ext = 400e-9, nm_ext = 500e-9, incr = 1,
              low = 0.4, riu_set = 'no', riu_cond = 3, swt_set = 'no', swt_cond = 0.3, verb = 0):
    
    ### Function that explores layer width combination for generating band gaps with "deep"
    ### Bloch surface waves, saving the best combinations. n_list is the initial list of 
    ### indexes, and pol ('rs' or 'rp') represents the polarization of the light, 
    ### steps is the number of steps that the matrix_multilayer function will go through, 
    ### change represents whether the angle ('a') or wavelength ('w') are changing. 
    ### a_i/a_f are the initial/final angles for changing angle, w_c being the constant 
    ### wavelength for those. w_i/w_f are the initial/final wavelengths for changing 
    ### wavelength, a_c being the constant angle.
    ### def_ext represents up to how much depth can be added to the defect, and nm_ext 
    ### represents how much depth can be added to the non-defect layers. incr is how many
    ### nanometers the code will jump through for each iteration.
    ### The total number of runs should be (def_ext * nm_ext * nm_ext) / incr ** 3.
    ### Returns, in order, a list with lists of widths for the layer, a list with values for
    ### the minimums observed, and the minimum value observed, the indexes matching for
    ### all of them. 
    ### If riu_set/swt_set are initialized as 'yes', the code will also filter out multilayers
    ### based on their RIU/SWT values. It will first check if a multilayer achieves the 
    ### minimum desired, and then it will check the RIU/SWT minimum conditions. 
    ### which are set by riu_cond and swt_cond. If RIU/SWT are turned on and
    ### verbosity is 1 or 2, the code will also give the calculated RIU/SWT
    ### values for the respective multilayers. 
    
    numbr_explr = (round_up((nm_ext*10e9 / incr) * (nm_ext*10e9 / incr) * (def_ext*10e9 / incr)), 0)
    print("A total of " + str(numbr_explr) + " layers expected to be explored.")
    print("Starting incremental parameter exploration...\n")
    
    d_set = []
    wv_p = []
    dist_n = []
    dist = steps
    n1 = n_list[1]
    n2 = n_list[2]
    nm_ext = int(nm_ext * 10e8)
    def_ext = int(def_ext * 10e8)

    if change == 'a':
        d1 = w_c / (8 * n1 * np.cos(a_i))
        d2 = w_c / (8 * n2 * np.cos(a_i))
        
    elif change == 'w':
        d1 = w_i / (8 * n1 * np.cos(a_c))
        d2 = w_i / (8 * n2 * np.cos(a_c))
        a_i = np.arcsin()
        
    else:
        print("Error, invalid angle/wavelength mode argument.")
        sys.exit() 
    
    timer = 0
    timer_d = 0
    
    init_timer = time.perf_counter()
    
    for ext_1 in range(0, nm_ext, incr):
        for ext_2 in range(0, nm_ext, incr):
            
            d = []
              
            for p in range(0, len(n_list)-2, 2):
            
                d.append(d1 + ext_1 * 10e-9)
                d.append(d2 + ext_2 * 10e-9)
            
            d[-1] = d[-1] * 1.2
                        
            for i in range(0, def_ext, incr):
                        
                d[-1] = d[-1] + incr * 10e-9
                                
                x, width, bot = bloch_wave(n_list, d, mode = pol, step = steps, option_aw = change, a1 = a_i,
                            a2 = a_f, w_t = w_c, w1 = w_i, w2 = w_f, ang = a_c, roof = 0.9,
                            minimal = 0.4, verb = 0)
                
                if (x != False):
                    if bot[0] < low:
                        if riu_set == 'yes':
                            riu_t = riu_calc(n_list, d, pol = pol, steps = steps, change = change, 
                                          a_i = a_i, a_f = a_f, w_c = w_c, w_i = w_i, w_f = w_f,
                                          a_c = a_c)
                            if riu_t < riu_cond:
                                continue
                            else:
                                riu_t = round_up(float(riu_t), 3)
                        if swt_set == 'yes':
                            swt_t = swt_calc(n_list, d, pol = pol, steps = steps, change = change, 
                                          a_i = a_i, a_f = a_f, w_c = w_c, w_i = w_i, w_f = w_f,
                                          a_c = a_c)
                            if swt_t < swt_cond:
                                continue
                            else:
                                swt_t = round_up(float(swt_t), 3)
                        if change == 'a':
                            for o in range(0, len(x)):
                                x[o] = round_up(float((x[o] / steps) + a_i) * 180/np.pi, 3)
                            for o in range(0, len(width)):
                                width[o] = round_up(float((width[o] / steps) + a_i) * 180/np.pi, 3)
                        elif change == 'w':
                            for o in range(0, len(x)):
                                x[o] = round_up(float((x[o] / steps) + w_i), 3)
                            for o in range(0, len(width)):
                                width[o] = round_up(float((width[o] / steps) + w_i), 3)
                        for o in range(0, len(bot)):
                            bot[o] = round_up(float(bot[o]), 3)
                        d_set.append(copy.copy(d))
                        wv_p.append(copy.copy(x))
                        timer_d += 1
                        if verb == 1:
                            if (riu_set == 'yes') and (swt_set == 'yes'):
                                print("Layer " + str(len(d_set)) + 
                                  " appended with a reflectivity minimum of " 
                                  + str(bot[0]) + ".RIU value of " + str(riu_t) + 
                                  " and SWT value of " + str(swt_t) + ".") 
                            elif riu_set == 'yes':
                                print("Layer " + str(len(d_set)) + 
                                  " appended with a reflectivity minimum of " 
                                  + str(bot[0]) + ".RIU value of " + str(riu_t) + ".")
                            elif swt_set == 'yes':
                                print("Layer " + str(len(d_set)) + 
                                  " appended with a reflectivity minimum of " 
                                  + str(bot[0]) + ".SWT value of " + str(swt_t) + ".")
                            else:
                                print("Layer " + str(len(d_set)) + 
                                  " appended with a reflectivity minimum of " 
                                  + str(bot[0]))
                        if verb == 2:
                            if change == 'a':
                                print("Angle of minimum: " + str(x))
                                if (riu_set == 'yes'):
                                    print(str(riu_t) + " degrees per RIU.")
                                if (swt_set == 'yes'):
                                    print(str(swt_t) + " degrees per SWT.")
                            elif change == 'w':
                                print("Wavelength of minimum: " + str(x))
                                if (riu_set == 'yes'):
                                    print(str(riu_t) + " nanometers per RIU.")
                                if (swt_set == 'yes'):
                                    print(str(swt_t) + " nanometers per SWT.")
                            print("Minimum observed: " + str(bot))
                            print(str(timer_d) + "# multilayer setup appended.")
                            real_timer = np.round((time.perf_counter() - init_timer) * 100) / 100
                            print(str(real_timer) + " seconds elapsed.\n")
                del x, width, bot        
                timer += 1
                if timer % 50 == 0 and verb > 0:
                    real_timer = np.round((time.perf_counter() - init_timer) * 100) / 100
                    print(str(timer) + " runs in " + str(real_timer) + " seconds.\n")

    real_timer = np.round((time.perf_counter() - init_timer) * 100) / 100 
    print("Total time: " + str(real_timer) +  ".")
    print("Total steps: " + str(timer) + ".")
    print("A total of " + str(len(d_set)) + " multilayers obtained.")
    
    return d_set, wv_p

##########################################################################################

def round_up(n, decimals=0):
    ### boilerplate function to round up numbers for output
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier

##########################################################################################

### Boilerplates: 

def export(var_type, target_var, target_file, csv = False):
    
    if os.path.exists(target_file) == True:
        print("The file + \"" + target_file + "\" already exists.")
    elif csv == True:
        d = varDict[target_var][:,0]
        a = varDict[taret_var][:,1]
        x = input("What is the independent variable in this simulation? (i.e. \"Wavelength\")\n\n")
        y = input("What is the dependent variable in this simulation? (i.e. \"Reflectivity\"\n\n")
        fields = [x, y]
        to_save = []
        for i in range(0, len(d)):
            to_save.append([d[i], a[i]])
        fp = open(target_file, 'w')
        write = csv.writer(fp)
        write.writerow(fields)
        write.writerow(to_save)
        fp.close()
    elif var_type == 'static':
        to_save = varDict[target_var]
        fp = open(target_file, 'w')
        fp.write(to_save)
        fp.close()
    elif var_type == 'dynamic':
        to_save = dynamic_formulas[target_var]
        fp = open(target_file, 'w')
        fp.write(to_save)
        fp.close()
    else:
        print("Invalid input.")

###########################################################################################

def import_b(var_type, save_name, target_file, csv = False):

    if os.path.exists(target_file) == False:
        print("The file +\"" + target_file + "\" does not exist.")
    elif csv == True:
        fp = open(target_file, 'r')
        csvFile = csv.DictReader(fp)
        to_savex = []
        to_savey = []
        for line in csvFile:
            to_savex.append(line[0])
            to_savey.append(line[1])
        del to_savex[0]
        del to_savey[0]
        to_save = np.array(to_savex, to_savey)
        varDict[save_name] = to_save
        fp.close()
    elif var_type == 'static':
        fp = open(target_file, 'r')
        varDict[save_name] = fp.read()
        fp.close()
    elif var_type == 'dynamic':
        fp = open(target_file, 'r')
        varDict[save_name] = fp.read()
        fp.close()

###########################################################################################

def crit_ang(n_list):
    return np.arcsin(n_list[-1] / n_list) - 0.05

###########################################################################################

def is_float(element):
    try:
        float(element)
        return True
    except ValueError:
        return False

###########################################################################################

### Help text:

while True:

    start = '''
    Hello, this is a script that runs through the various functionalities of the Optical
    Multilayer module, written by David C. Heson, Jack Liu, and Dr. Bill Robertson over the
    duration of the 2022 Middle Tennessee University Computational Sciences (R)esearch 
    (E)xperience for (U)ndergraduates, funded by the (N)ational (S)cience (F)oundation.
    This code uses the matrix formulation written by [INSERT PAPER NAME HERE] to calculate
    Fresnel coefficients of reflection/transmission, and older Fortran/MATLAB code originally
    written by Dr. Robertson to calculate the electrical field profile accross an optical multilayer.
    
    This script provides an interactive environment to use the module's functions. Inputs such as
    the width of the layers or the indexes of refraction are input and then made into objects which
    can be easily used over and over within the code. 
    
    Familiarizing yourself with how the \"input\", \"multilayer sim\", and \"graph\" commands work is
    a good place to start.
    
    This code is designed for Python 3.x, using the Numpy and Matplotlib Python modules. All commands
    are case sensitive.
    
    At times, \"exit\" commands will return an error due to the way loop-breaking is managed in specific cases.
    The code executed properly if this happens, the messaging will be improved in later versions.
    
    For any questions, suggestions, or concerns, please reach out on the GitHub page for this project or 
    to the code maintainer, David C. Heson (email: dch376@msstate.edu).
    '''
    
    loop_start = '''
    Please input an appropiate command for the code to execute. 
    Input \"help\" for a list of commands.\n
    '''
    
    help_text = '''
    This is the list of the various commands within the script. To get more information about
    a specific command, run the command and then run help within it. (i.e., to get information about
    the \"graph\" command, input \"graph\", and then input \"help\"). To exit out of an action input
    \"exit\", which can also be used to stop the script.
    
    Basic Functionalities:
    
    - export
    - export data
    - import
    - import data
    - print
    - input
    - delete
    - resolution
    - hyperparameters
    - multilayer sim
    - efield sim
    - bloch detector
    - graph
    - graph efield
    - riu calculator
    - swt calculator
    - reset
    
    Advanced Functionalities:
    
    - multilayer sim dynamic
    - dynamic formulas
    - dynamic index
    - explore
    '''
    
    back_main = '''
    \nExiting current module, going back to main.
    '''
    
    back_sub = '''
    \nExiting current subroutine, going back to previous function.
    '''
    
    exception_gen = '''
    \nException occured! Returning to main...
    '''
    
    no_help = '''
    No additional information available for this module.
    '''
    
    invalid_command = '''
    Invalid command. Please input a valid command.
    '''
    
    print_help = '''
    Subroutine that prints out a specific variable. If you have a variable that is a list or array
    you can input \"index\" for the first command, which will then prompt you to give the 
    variable name, and the index of the element within the variable which you want to print.
    If you want to see the name of all variables currently stored, input \"all"\.
    To print the saved dynamic index formulas, input \"dynamic formulas\" at first.
    '''
    
    resolution_help = '''
    Changes the resolution at which the runs throughout the system are made. This is equivalent to
    how many points are simulated for each run over the existing interval. The lower the resolution,
    the faster the code will run (to a significant ammount), however it will also output rougher results,
    higher resolutions being required to accurately capture phenomenas such as Bloch Surface Waves.
    '''
    
    input_help = '''
    Function to create variables to be used in simulating multilayers. All inputs should be of the
    form real_part, imaginary_part, but it should be fine if only the real part is passed (assuming 0 for
    the imaginary part). For example, for inputing the index for glass, the input can be as simple
    as \"1.5\". For SiO2, the input would be \"1.46+3.4e-5\. Type in "demo" to get a premade set of parameters
    \"n_demo\" (for the index) and \"d_demo"\ (for the depths) that you can use. 
    Make sure there is no whitespace in the input!
    '''
    
    hyper_help = '''
    Function to adjust the hyperparameters used accross the various simulations of light passing
    through a multilayer. The following are the adjustable hyperparameter commands:
    - resolution
    - iterate
    - polarization
    - fixed angle
    - fixed wavelength
    - range
    - efield limit 
    - minimum required
    - exploration sens
    '''
    
    crit_ang_help = '''
    Using a critical angle initialization can allow better resolution for the area of total internal
    reflection, where Bloch Surface Waves occurs. This critical angle will be called for all Bloch
    Surface Wave related operations if \"yes"\ is input, a critical angle calculated as the arcsine of the 
    index of the bottom layer divided by the index of the top layer, everything minus 0.05 radians will be 
    used. The 0.05 radian offset is given so that there is enough "empty space" for the Bloch Surface Wave 
    detector to operate properly if a BSW is present very close to the critical angle. 
    '''
    
    iterate_help = '''
    Changes over which parameter the various functions will iterate through. Can iterate over 
    either \"wavelength\" or \"angle\".
    '''
    
    exploration_sens_help = '''
    to_do
    '''
    
    efield_help = '''
    Simulates the electric field accross the specified multilayer (using stored variables for 
    indexes and widhts). Uses the angle and wavelength hyperparameters, the simulation moving 
    from top to bottom through the multilayers, and then going outside the last layer for 
    the hyperparameter \"efield_limit\" distance. Option to graph and save the simulation. 
    The graph for the electrical field simulation will also include vertical lines that denote the 
    physical boundaries for each layer. 
    '''
    
    graph_help = '''
    Graphing subroutine that graphs any stored reflectivity or transmission simulations. 
    Provides capabilities to customize the way the graph is made, and include multiple 
    simulations on the same graph. 
    '''
    
    graph_efield_help = ''' 
    Graphing subroutine specifically for graphing stored electrical field simulations. It requires a specified 
    depths variable, since it will insert lines to denote the borders between multilayers. Provides capabilities
    to customize the way the graph is made, and include multiple simulations on the same graph. 
    '''
    
    dynamic_ind_help = '''
    Subroutine that creates an ordered list with the key list that will be dispatched for simulating an optical
    multilayer with wavelength dependent indexes of refraction.
    '''
    
    dyn_for_help = '''
    Subroutine that creates formulas for dynamic indexing. Inputs are saved to a dictionary that is automatically passed
    to the \"multilayer sim dynamic subroutine\". For input, if you wish for one of the mediums to have a static index, 
    simply input whatever static index it may be. For example, for glass the formula input would just be \"1.5\".
    For actual formulas, the input should be in the form that you'd normaly type formulas in Python code.
    Input example for TiO2 dynamic index formula (the formula assumes wavelength to be in microns):
    np.sqrt((5.913+(0.2441)/(-0.0803+(x*10**6)**2))+0.0007j
    '''
    
    bloch_help = '''
    Subroutine to analyze a given multilayer setup and determine if it exhibits a Bloch Surface Wave (without graphing it).
    While the back-end function for the Bloch Surface Wave detector is useful in other subroutines, for manually checking
    if a specific multilayer does exhibit a Bloch Surface Wave, the Electrical Field Simulator (\"efield sim\") might be 
    more useful, since it guarantees that what is observed is actually BSW and not just interference. However, \"bloch detector\" 
    does return the reflectivity minimums detected as Bloch Surface Waves, and the angle/wavelength they occur at.
    '''
    
    riu_help = '''
    Subroutine which calculates the degrees or nanometers per (R)efractive (I)ndex (U)nits, given a multilayer. It is reccomended
    to run this at a resolution as high as possible, due to the chance that the altered multilayer setup used to obtain the RIU value
    may exhibit a Bloch Surface Wave that's extremely tight. The calculation is made by changing the final layer index by 0.01, and then
    comparing the position of the BSW of the original multilayer design with that of the altered multilayer design. Will try to use the 
    critical angle functionality if it is triggered on. 
    '''
    
    swt_help = '''
    Subroutine which calculates the degrees or nanometers per (S)hift (W)ith (T)hickness, given a multilayer. It is reccomended
    to run this at a resolution as high as possible, due to the chance that the altered multilayer setup used to obtain the SWT value
    may exhibit a Bloch Surface Wave that's extremely tight. The calculation is made by changing the final layer thickness by 10 nanometers, 
    and then comparing the position of the BSW of the original multilayer design with that of the altered multilayer design. Will try to 
    use the  critical angle functionality if it is triggered on. 
    '''
    
    explore_help = '''
    Subroutine which explores multilayer setups within a given range of parameters, and calculating the minimum values observed by the BSW 
    detector function. Found results could display dips in reflectivity due to interference conditions, and not the presence of a BSW, hence 
    any obtained layers should be confirmed using the electric field graphing functions. The found layers will be stored in the variable dictionary
    as \"insertedname_#\". For example, if the run name is \"test\" and 5 multilayers are found, the stored layers will be named \"test_1\", \"test_2\", 
    \"test_3"\, "\test_4"\, and \"test_5\". If the RIU and/or the SWT filters are chosen, the code will also filter out multilayers that exhibit 
    RIU\SWT measures below the values indicated in the hyperparameters under \"sens threshold\". 
    '''
    
    export_help = '''
    Simple tool to export data or a variable created through the script for future use. It will create a .txt document that simply has the respective information.
    For exporting data as a .csv, use the \"export data\" command specifically. 
    '''
    
    import_help = '''
    Simple tool to import data or a variable. Works well with variables created within the script, and a variable could also be potentially defined outside of the
    script and then imported this way.
    For importing data from a .csv, use the \"import data\" command specfiically. 
    '''
    
    export_data_help = '''
    Subroutine of the export tool to export data specifically as .csv. The names input will be used as column headers.
    '''
    
    import_data_help = '''
    Subroutine of the import tool to import .csv data. Will follow the data format from the export_data subroutine, meaning
    that it will specifically look at all the rows below the column headers, taking the first column as the independent variable
    and the second column as the dependent variable. 
    '''
    
    input_alert = '''
    It is STRONGLY suggested to see the documentation for this command before
    using it! Type \"ok\" when asked for a variable name to disable this alert for this session.
    '''
    
    input_demo = '''
    Pre-made variables for index (\"n_demo\") and layer depths (\"d_demo\") created.
    '''
    
    input_autobuild = '''
    Do you want to use the multilayer autobuilder? It will generate either an index list or width list
    that corresponds to a regular multilayer design. Type \"exit\" to quit the loop anytime. (yes/no)\n\n
    '''
    
    polarization_int = '''
    Please insert which polarization / output you want to be used throughout the program.
    The options are:
    - \"rs\" (reflection with S-polarization)
    - \"rp\" (reflection with P-polarization)
    - \"ts\" (transmission with S-polarization)
    - \"tp\" (transmission with P-polarization)
    '''
    
    hyper_int = '''
    Please insert which hyperparameter you wish to change. Alternatively, type in \"check\" to get a 
    full list of all hyperparameters.
    '''
    
    reset_int = '''
    Do you wish to reset all the hyperparameters (\"hyper\"), all the variables (\"var\") or everything (\"all\").
    Type \"exit\" to not reset anything.
    '''
    
    break

### Variable set-up:

def initializing_hyperparameters():
    global resolution 
    resolution = 5000
    global iterate 
    iterate = "angle"
    global fix_angle 
    fix_angle = 60 * np.pi/180
    global start_angle 
    start_angle = 0
    global end_angle 
    end_angle = np.pi/2
    global fix_wave 
    fix_wave = 635e-9
    global start_wave 
    start_wave = 400e-9
    global end_wave 
    end_wave = 750e-9
    global pol_mode 
    pol_mode = 'rs'
    global efield_limit 
    efield_limit = 1000e-9
    global min_required
    min_required = 0.4
    global crit_ang_set
    crit_ang_set = 'no'
    global first_angle
    first_angle = 0
    global riu_set
    riu_set = 'no'
    global swt_set
    swt_set = 'no'
    global riu_cond
    riu_cond = 3
    global swt_cond
    swt_cond = 0.03
    global set_low
    set_low = 0.4

def initializing_variables():
    global varDict 
    varDict = {}
    global dynamic_formulas 
    dynamic_formulas = {}
    
initializing_hyperparameters()
initializing_variables()
alert = 'yes'
print(start)

### Main code loop:

while True:

    command = input(loop_start)

    ##########################
    
    if command == 'exit':
        print("Have a good day!  ʕ •ᴥ• ʔ")
        break

    ##########################
    
    elif command == 'help':
        print(help_text)
        print(back_main)
 
    ##########################
    
    elif command == 'export':
        try:
            while True:
                varName = input("Please specify the name of the variable/data which you wish to export.\n\n")
                if varName == 'exit':
                    print(back_main)
                    break
                elif varName == 'help':
                    print(export_help)
                    continue
                varType = input("Is the variable to be saved a generated dynamic formula? (yes/no)\n\n")
                if varType == 'exit':
                    print(back_main)
                    break
                elif varName == 'help':
                    print(export_help)
                    continue
                if varType == 'yes':
                    varType == 'dynamic'
                    if varName in dynamic_formulas == False:
                        print("Variable " + varName + " not found in the dynamic variable dictionary.")
                        print(back_main)
                        break
                elif varType == 'no':
                    varType == 'static'
                    if varName in varDict == False:
                        print("Variable " + varName + " not found in the variable dictionary.")
                        print(back_main)
                        break
                print("Please specify the name of the file where you want to save this.")
                file_save = input("Alternatively, input \" \" to save it to the folder where this script is in with the name of the variable.\n\n")
                if file_save == " ":
                    export(var_type = varType, target_var = varName, target_file = varName)
                else:
                    export(var_type = varType, target_var = varName, target_file = file_save)
                print("\"" + varName + "\" succesfully exported!")
                print(back_sub)	
        except:
            print(exception_gen)
    
    ##########################  
    
    elif command == 'export data':
        try:
            while True:
                varName = input("Please input the name of the variable under which the data you wish to save is stored.\n\n")
                if varName == 'exit':
                    print(back_main)
                    break
                elif varName == 'help':
                    print(export_data_help)
                    continue
                elif varName not in varDict:
                    print(str(to_graph), " not found in stored in variables.")
                    continue
                print("Please specify the name of the file where you want to save this.")
                file_save = input("Alternatively, input \" \" to save it to the folder where this script is in with the name the data is saved under.\n\n")
                if file_save == " ":
                    file_save = varName + '.csv'
                    export(var_type = varType, target_var = varName, target_file = file_save)
                else:
                    export(var_type = varType, target_var = varName, target_file = file_save)
                print("\"" + varName + "\" succesfully exported!")
                print(back_sub)	
        except:
            print(exception_gen)
            
    ##########################

    elif command == 'import data':
        try:
            while True:
                varType = 1
                varName = input("Please input the name under which you want to save the imported data.\n\n")
                if varName == 'exit':
                    print(back_main)
                    break
                elif varName == 'help':
                    print(import_data_help)
                    continue
                elif varName not in varDict:
                    print(str(to_graph), " not found in stored in variables.")
                    continue
                print("Please specify the name of the file where you want to save this.")
                file_save = input("Alternatively, input \" \" to save it to the folder where this script is in with the name the data is saved under.\n\n")
                if file_save == " ":
                    import_b(var_type = varType, target_var = varName, target_file = varName)
                else:
                    import_b(var_type = varType, target_var = varName, target_file = file_save)
                print("\"" + varName + "\" succesfully exported!")
                print(back_sub)	
        except:
            print(exception_gen)
    
    ##########################
    
    elif command == 'import':
        try:
            while True:
                varName = input("Please specify the name you want to import the variable/data as.\n\n")
                if varName == 'exit':
                    print(back_main)
                    break
                elif varName == 'help':
                    print(import_help)
                    continue
                varType = input("Is the variable/data you are importing a dynamic formula? (yes/no)\n\n")
                if varType == 'exit':
                    print(back_main)
                    break
                elif varType == 'help':
                    print(import_help)
                    continue
                elif varType == 'yes':
                    varType = 'dynamic'
                elif varType == 'no':
                    varType = 'static'
                print("Please specify the name of the file from where the data is being imported.")
                varFile = input("Alternatively, input \" \" if it is saved with the name you indicated and it is in the same folder as this script.\n\n")
                if varFile == " ":
                    import_b(varType, varName, varName)
                else:
                    import_b(varType, varName, varFile)
                print("\"" + varName + "\" succesfully imported!")
                print(back_sub)
        except:
            print(exception_gen)
    
    ##########################
    
    elif command == 'import data':
        try:
            while True:
                varType = 1
                varName = input("Please specify the name you want to import the variable/data as.\n\n")
                if varName == 'exit':
                    print(back_main)
                    break
                elif varName == 'help':
                    print(import_help)
                    continue
                print("Please specify the name of the file from where the data is being imported.")
                varFile = input("Alternatively, input \" \" if it is saved with the name you indicated and it is in the same folder as this script.\n\n")
                if varFile == " ":
                    varFile = varName + '.csv'
                    import_b(varType, varName, varFile, csv = True)
                else:
                    import_b(varType, varName, varFile, csv = True)
                print("\"" + varName + "\" succesfully imported!")
                print(back_sub)				
        except:
            print(exception_gen)
    
    ##########################
    
    elif command == 'print':
        try:
            while True:
                print("Type the name of the variable which you want to see.")
                command = input("To see all stored variables, type \"all\". \n\n")
                if command == "exit":
                    print(back_main)
                    break
                elif command == "help":
                    print(print_help)
                    continue
                elif command == "all":
                    print("The following are all the variables currently stored in this run: ")
                    for variable in varDict:
                        print(str(variable))
                elif command == 'index':
                    el = input("Type the name of the variable for which you want to see an element of.\n\n")
                    index = int(input("Type the index of the element which you want to see.\n\n"))
                    el = varDict[el]
                    print(str(el[index]), "\n")
                    print("Variable printed.")
                elif command == 'dynamic formulas':
                    print("The following are all the defined dynamic index formulas for this run: ")
                    for variable in dynamic_formulas:
                        print(str(variable))
                else:    
                    print(str(varDict[command]), "\n")
                    print("Variable printed.")
        except:
            print(exception_gen)
    
    ##########################
    
    elif command == 'input':
        try:
            if alert == 'yes':
                print(input_alert)
            while True:
                var_name = input("Please specify the variable name for your input.\n\n")
                if var_name == 'exit':
                    print(back_main)
                    break
                if var_name == 'ok':
                    print("Alert disabled.")
                    alert == 'no'
                if var_name in varDict:
                    while var_name in varDict:
                        try:
                            print("Variable ", str(var_name), " already exists. Please enter a different name.")
                            var_name = input()
                            if var_name == "exit":
                                raise exit
                        except exit:
                            print(back_main)   
                elif var_name == "demo":
                    n_demo = np.array([complex(1.5, 0), complex(2.3, 1.5e-4), complex(1.46, 3.4e-5),
                                    complex(2.3, 1.5e-4), complex(1.46, 3.4e-5),
                                    complex(2.3, 1.5e-4), complex(1.46, 3.4e-5),
                                    complex(2.3, 1.5e-4), complex(1.46, 3.4e-5), complex(1.33)])
                    varDict["n_demo"] = np.copy(n_demo)
                    d_demo = np.array([complex(104.8e-9, 0), complex(161.8e-9, 0),
                                    complex(104.8e-9, 0), complex(161.8e-9, 0),
                                    complex(104.8e-9, 0), complex(161.8e-9, 0),
                                    complex(104.8e-9, 0), complex(280.0e-9, 0)])
                    varDict["d_demo"] = np.copy(d_demo)
                    del n_demo, d_demo
                    print(input_demo)
                    break
                else:
                    var_list = []
                    command = input(input_autobuild)
                    if command == 'yes':
                        mode = input("Index or depth variable?\n\n")
                        if mode == "exit":
                            print(back_sub)
                            break
                        elif mode == 'index':
                            layer_ind = []
                            top = input("Please enter the index for the top medium.\n\n")
                            if top == "exit":
                                print(back_sub)
                                break
                            top = complex(top)
                            bot = input("Please enter the index for the bottom medium.\n\n")
                            if bot == "exit":
                                print(back_sub)
                                break
                            bot = complex(bot)
                            period = input("Please enter the periodicity of the multilayer.\n\n")
                            if period == "exit":
                                print(back_sub)
                                break
                            period = int(period)
                            pairs = input("Please enter how many sets of layers are in the multilayer.\n\n")
                            if pairs == 'exit':
                                print(back_sub)
                                break
                            pairs = int(pairs)
                            for i in range(1, period + 1):
                                print("Please enter the index for layer ", str(i), ".")
                                index_toapp = input()
                                if index_toapp == "exit":
                                    print(back_sub)
                                    break
                                else:
                                    index_toapp = complex(index_toapp)
                                    layer_ind.append(index_toapp)
                            var_list.append(top)
                            for i in range(0, pairs):
                                for s in range(0, period):
                                    var_list.append(layer_ind[s])
                            var_list.append(bot)
                            varDict[var_name] = np.copy(np.array(var_list))
                            print("Index variable ", str(var_name), " succesfully created!")
                            del(var_list)
                            print(back_sub)
                            
                        elif mode == 'depth':
                            layer_w = []
                            period = input("Please enter the periodicity of the multilayer.\n\n")
                            if period == "exit":
                                print(back_sub)
                                break
                            period = int(period)
                            pairs = input("Please enter how many sets of layers are in the multilayer.\n\n")
                            if pairs == 'exit':
                                print(back_sub)
                                break
                            pairs = int(pairs)
                            for i in range(1, period + 1):
                                print("Please enter the depth of layer ", str(i), ".")
                                depth_toapp = input()
                                if depth_toapp == "exit":
                                    print(back_sub)
                                    break
                                else:
                                    depth_toapp = complex(depth_toapp)
                                    layer_w.append(depth_toapp)
                            defect = input("Please enter the depth of the defect layer.\n\n")
                            if defect == "exit":
                                print(back_sub)
                                break
                            defect = complex(defect)
                            pos_def = input("Please enter which layer is the defect layer (\"-1\" defaults to the last layer).\n\n")
                            if pos_def == "exit":
                                print(back_sub)
                                break
                            pos_def = int(pos_def)
                            for i in range(0, pairs):
                                for s in range(0, period):
                                    var_list.append(layer_w[s])
                            var_list[pos_def] = defect
                            varDict[var_name] = np.copy(np.array(var_list))
                            print("Layer depth variable ", str(var_name), " succesfully created!")
                            print(back_sub)
                            
                        else:
                            print("You will be prompted to input values for your variable. Type \"done\" to finish it.")
                            print("Typing \"exit\" WILL not save your variable, and just force quit the sequence.")
                            while True:
                                var = input("Please input a value for your variable.\n\n")
                                if var == "exit":
                                    print(back_sub)
                                    break
                                elif var == "done":
                                    varDict[var_name] = np.copy(np.array(var_list))
                                    print("Variable ", var_name, " sucesfully created with the given input.")
                                    break
                                else:
                                    var_list.append(complex(var))
                    elif command == "exit":
                        print(back_sub)
                        break
                    else:
                        print("Please input a valid command.")
        except:
            print(exception_gen)
    
    ##########################
    
    elif command == 'delete':
        try:
            while True:
                command = input("Please specify which variable you wish to delete.\n\n")
                if command == "exit":
                    print(back_main)
                    break
                elif command == "help":
                    print(delete_help)
                    continue
                del varDict[command]
                print('Variable ', str(command), 'sucesfully deleted.')
        except:
            print(exception_gen)

    ##########################
    
    elif command == 'hyperparameters':
        try:
            while True:
                command = input(hyper_int)
                if command == 'exit':
                    print(back_main)
                    break
                elif command == 'help':
                    print(hyper_help)
                    continue
                elif command == 'check':
                    print("These are the current hyperparameters: ")
                    print("Resolution: " + str(resolution))
                    print("Polarization mode: " + str(pol_mode))
                    print("Iterate over: " + str(iterate))
                    print("Starting angle: " + str(start_angle))
                    print("Ending angle: " + str(end_angle))
                    print("Fixed angle: " + str(fix_angle))
                    print("Starting wavelength: " + str(start_wave))
                    print("Ending wavelength: " + str(end_wave))
                    print("Fixed wavelength: " + str(fix_wave))
                    print("Minimum required for Bloch: " + str(min_required))
                    print("Minimum RIU value for explore: " + str(riu_cond))
                    print("Minimum SWT value for explore: " + str(swt_cond))
                    print("Required reflectivity minimum for explore: " + str(set_low))
                elif command == 'exploration low':
                    while True:
                        command == input("Please insert how low should the detected reflectivity dips be for the explore function.\n\n")
                        if command == 'exit':
                            print(back_sub)
                            break
                        elif command == 'help':
                            print(no_help)
                            continue
                        elif is_float(command) == False:
                            print(invalid_command)
                            continue
                        else:
                            set_low = command
                            print(back_sub)
                            break
                elif command == 'exploration sens':
                    while True:
                        command == input("Which precision threshold do you wish to adjust? (riu/swt)\n\n")
                        if command == 'exit':
                            print(back_sub)
                            break
                        elif command == 'help':
                            print("Change at what level of RIU or SWT precision the \"explore\" function will select multilayer designs.")
                            continue
                        elif command == 'swt':
                            command == input("Please insert what SWT precision you wish to use for the multilayer explore.")
                            while True:
                                if command == 'exit':
                                    print(back_sub)
                                    break
                                elif command == 'help':
                                    print(no_help)
                                    continue
                                elif is_float(command) == True:
                                    swt_cond = float(command)
                                    print("SWT precision condition updated to " + str(swt_cond) + ".")
                                    break
                                else:
                                    print(invalid_command)
                        elif command == 'riu':
                            command == input("Please insert what RIU precision you wish to use for the multilayer explore.")
                            while True:
                                if command == 'exit':
                                    print(back_sub)
                                    break
                                elif command == 'help':
                                    print(no_help)
                                    continue
                                elif is_float(command) == True:
                                    swt_cond = float(command)
                                    print("RIU precision condition updated to " + str(riu_cond) + ".")
                                    break
                                else:
                                    print(invalid_command)
                        else:
                            print(invalid_command)
                elif command == 'critical angle':
                    print("Do you wish to use a calculated critical angle for the starting parameter for Bloch Surface Wave-related operations? (yes/no)")
                    while True:
                        command = input()
                        if command == 'exit':
                            print(back_sub)
                            break
                        elif command == 'help':
                            print(crit_ang_help)
                        elif command == 'yes' or command == 'no':
                            crit_ang_set = command
                            print("Critical angle option succesfully changed to " + crit_ang_set + "radians.")
                            print(back_sub)
                            break
                        else:
                            print(invalid_command)
                elif command == 'min required':
                    while True:
                        command = input("Please input what minimum reflection coefficient you wish to use for Bloch Surface Wave detection.\n\n")
                        if command == 'exit':
                            print(back_sub)
                            break
                        elif command == 'help':
                            print(no_help)
                        elif is_float(command) == True:
                            min_required = float(command)
                            print("Minimum reflection coefficient required for Bloch Surface Wave detection set to " + str(min_required) + ".")
                            print(back_sub)
                            break
                        else:
                            print(invalid_command)
                elif command == 'iterate':
                    while True:
                        command = input("Please input whether you want to change the code to iterate over angle or over wavelength.\n\n")
                        if command == "exit":
                            print(back_sub)
                            break
                        elif command == 'help':
                            print(iterate_help)
                        elif command == 'angle' or command == 'wave':
                            iterate = command
                            print("Change parameter succesfully updated to be ", str(command), ".")
                            print(back_sub)
                            break
                        else:
                            print(invalid_command)
                elif command == 'resolution':
                    while True:
                        command = input("Please provide a value for the resolution you desire to use.\n\n")
                        if command == 'exit':
                            print(back_sub)
                            break
                        elif command == 'help':
                            print(resolution_help)
                        elif is_float(command) == True:
                            resolution = int(command)
                            print("Resolution succesfully changed to ", str(command), " steps.")
                            break
                        else:
                            print(invalid_command)
                elif command == "polarization":
                    while True:
                        command = input(polarization_int)
                        if command == "exit":
                            print(back_sub)
                            break
                        elif command == "help":
                            print(no_help)
                        elif command == 'rs' or command == 'rp' or command == 'ts' or command == 'tp':
                            pol_mode = commannd
                            print("Polarization mode succesfully updated to ", str(command), ".")
                            print(back_sub)
                            break
                        else:
                            print(invalid_command)
                elif command == "fixed angle":
                    while True:
                        command = input("Please insert your new fixed angle (in radians).\n\n")
                        if command == 'exit':
                            print(back_sub)
                            break
                        elif command == 'help':
                            print(no_help)
                        elif is_float(command) == True:
                            fix_angle = float(command)
                            print("Fixed angle succesfully updated to " + fix_angle + " radians.")
                            print(back_sub)
                            break
                        else:
                            print(invalid_command)
                elif command == "fixed wavelength":
                    while True:
                        command = input("Please insert your new fixed wavelength (in meters).\n\n")
                        if command == 'exit':
                            print(back_sub)
                            break
                        elif command == 'help':
                            print(no_help)
                        elif is_float(command) == True:
                            fix_wave = command
                            print("Fixed wavelenght succesfully updated to " + fix_wave + " meters.")
                            print(back_sub)
                            break
                        else:
                            print(invalid_command)
                elif command == 'range':
                    while True:
                        var = input("What range variable do you want to adjust? (angle/wave)\n\n")
                        if var == 'exit':
                            print(back_sub)
                            break
                        elif var == 'help':
                            print(no_help)
                        elif var == 'angle':
                            while True:
                                var = input("Which angle do you want to adjust? (start/end)\n\n")
                                if var == 'exit':
                                    print(back_sub)
                                    break
                                elif var == 'help':
                                    print(no_help)
                                elif var == 'start':
                                    var = input("Please insert your new starting angle (in radians).\n\n")
                                    if var == "exit":
                                        print(back_sub)
                                        break
                                    elif var == 'help':
                                        print(no_help)
                                    elif is_float(var) == True:
                                        var = float(var)
                                        start_angle = var
                                        print("Starting angle sucesfully changed to ", str(start_angle), ".")
                                    else:
                                        print(invalid_command)
                                elif var == 'end':
                                    var = input("Please insert your new ending angle (in radians).\n\n")
                                    if var == "exit":
                                        print(back_sub)
                                        break
                                    elif var == 'help':
                                        print(no_help)
                                    elif is_float(command) == True:
                                        var = float(var)
                                        end_angle = var
                                        print("Ending angle succesfully changed to ", str(end_angle), ".")
                                    else:
                                        print(invalid_command)
                                else:
                                    print(invalid_command)
                        elif var == 'wave':
                            while True:
                                var = input("Which wavelength do you want to adjust? (start/end)\n\n")
                                if var == 'exit':
                                    print(back_sub)
                                    break
                                elif var == 'start':
                                    var = input("Please insert your new starting wavelength (in nanometers)\n\n")
                                    if var == 'exit':
                                        print(back_sub)
                                        break
                                    elif var == 'help':
                                        print(no_help)
                                    elif is_float(command) == True:
                                        var = float(var)
                                        start_wave = var
                                        print("Starting wavelength sucesfully changed to ", str(start_wave), ".")
                                    else:
                                        print(invalid_command)
                                elif var == "end":
                                    var = input("Please insert your new ending wavelength (in nanometers).\n\n")
                                    if var == "exit":
                                        print(back_sub)
                                        break
                                    else:
                                        var = float(var)
                                        end_wave = var
                                        print("Ending wavelength sucesfully changed to ", str(end_wave), ".")
                                else:
                                    print(invalid_command)
                        else:
                            print(invalid_command)
                else:
                    print(invalid_command)
        except:
            print(exception_gen)
    
    ##########################
    
    elif command == 'multilayer sim':
        try:
            while True:
                print("Simulation of light passing through a multilayer.")
                n_list = input("Please type in which variable corresponds to the index list you want to use.\n\n")
                if n_list == "exit":
                    print(back_main)
                    break
                elif n_list not in varDict:
                        try:
                            while n_list not in varDict:
                                print(str(n_list), " not found in stored variables.")
                                n_list = input("Please input another variable name for the indexes to use.\n\n")
                                if n_list == 'exit':
                                    raise exit
                        except exit:
                            print(exception_gen)
                d_list = input("Please type in which variable corresponds to the depth list you want to use.\n\n")
                if d_list == "exit":
                    print(back_main)
                    break
                elif d_list not in varDict:
                            try:
                                while d_list not in varDict:
                                    print(str(to_graph), " not found in stored variables.")
                                    d_list = input("Please input another variable for the depth list to use.\n\n")
                                    if d_list == 'exit':
                                        raise exit
                            except exit:
                                print(exception_gen)
                print("Running simulation with given parameters and hyperparameters......")
                if iterate == "angle":
                    temp_aw = "a"
                elif iterate == "wavelength":
                    temp_aw = "w"
                result = multilayer(varDict[n_list], varDict[d_list], mode = pol_mode, step = resolution, 
                        option_aw = temp_aw, a1 = start_angle, a2 = end_angle, w_t = fix_wave,
                        w1 = start_wave, w2 = end_wave, ang = fix_angle)
                del temp_aw
                print("Done!")
                command = input("Do you wish to graph the result? (yes/no)\n\n")
                if command == "yes":
                    label = []
                    results = []
                    if iterate == "angle":
                        text = (str(fix_wave) + " nm")
                    elif iterate == "wavelength":
                        text = str(fix_angle + " degrees")
                    label.append(text)
                    results.append(result)
                    graph(results, label)
                elif command == "exit":
                    print(back_main)
                    break   
                command = input("Do you wish to save the data from the generated simulation? (yes/no)\n\n")
                if command == "yes":
                    var = input("Please type in to what variable you'd like to save the result.\n\n")
                    if var == "exit":
                        print(back_main)
                        break
                    else:
                        varDict[var] = copy.copy(result)
                        print("Result sucesfully generated and stored in ", var, ".")
                elif command == "exit":
                    print(back_main)
                print("Returning to input for mutilayer simulation. Type \"exit\" to return to main.")
        except:
            print(exception_gen)
    
    ##########################
    
    elif command == 'reset':
        try:
            while True:
                print(reset_int)
                command = input()
                if command == 'exit':
                    print(back_main)
                    break
                elif command == 'hyper':
                    initializing_hyperparameters()
                    print("Hyperparameters reset.")
                    print(back_main)
                    break
                elif command == 'var':
                    initializing_variables()
                    print("Variables wiped.")
                    print(back_main)
                    break
                elif command == 'all':
                    initializing_hyperparameters()
                    initializing_variables()
                    print("Hyperparameters reset and variables wiped.")
                    print(back_main)
                    break
                else:
                    print(invalid_command)
        except:
            print(exception_gen)
    
    ##########################
    
    elif command == 'riu calculator':
        try:
            print("RIU Calculator.")
            while True:
                n_list == input("Please insert the name of the index variable you wish to use for this multilayer.\n\n")
                if n_list == 'exit':
                    print(back_main)
                    break
                elif n_list == 'help':
                    print(riu_help)
                    continue
                while n_list not in varDict:
                    try:
                        print(str(n_list) + "variable not found in stored variables.")
                        n_list = input("Please insert another name for an index variable.\n\n")
                        if n_list == 'exit':
                            raise exit
                    except exit:
                        print(back_main)
                d_list = input("Please insert the name of the width variable you wish to use for this multilayer.\n\n")
                if d_list == 'exit':
                    print(back_main)
                    break
                elif d_list == 'help':
                    print(riu_help)
                    continue
                while d_list not in varDict:
                    try:
                        print(str(d_list) + "variable not found in stored variables.")
                        d_list = input("Please insert another name for the width variable.\n\n")
                        if d_list == 'exit':
                            raise exit
                    except exit:
                        print(back_main)
                if change == 'angle':
                    temp_aw = 'a'
                    if crit_ang_set == 'yes':
                        first_angle = crit_ang(varDict[n_list])
                    elif crit_ang_set == 'no':
                        first_angle = start_angle
                elif change == 'wave':
                    temp_aw = 'w'
                print("Running RIU calculator with given parameters and hyperparameters...")
                riu_val = riu_calc(varDict[n_list], varDict[d_list], pol_mode, resolution, change = temp_aw,
                                    a_i = first_angle, a_f = end_angle, w_c = fix_angle, w_i = start_wave, 
                                    w_f = end_wave, a_c = fix_angle, verb = 1)
        except:
            print(exception_gen)
            
    ##########################
    
    elif command == 'swt calculator':
        try:
            print("SWT Calculator.")
            while True:
                n_list == input("Please insert the name of the index variable you wish to use for this multilayer.\n\n")
                if n_list == 'exit':
                    print(back_main)
                    break
                elif n_list == 'help':
                    print(swt_help)
                    continue
                while n_list not in varDict:
                    try:
                        print(str(n_list) + "variable not found in stored variables.")
                        n_list = input("Please insert another name for an index variable.\n\n")
                        if n_list == 'exit':
                            raise exit
                    except exit:
                        print(back_main)
                d_list = input("Please insert the name of the width variable you wish to use for this multilayer.\n\n")
                if d_list == 'exit':
                    print(back_main)
                    break
                elif d_list == 'help':
                    print(swt_help)
                    continue
                while d_list not in varDict:
                    try:
                        print(str(d_list) + "variable not found in stored variables.")
                        d_list = input("Please insert another name for the width variable.\n\n")
                        if d_list == 'exit':
                            raise exit
                    except exit:
                        print(back_main)
                if change == 'angle':
                    temp_aw = 'a'
                    if crit_ang_set == 'yes':
                        first_angle = crit_ang(varDict[n_list])
                    elif crit_ang_set == 'no':
                        first_angle = start_angle
                elif change == 'wave':
                    temp_aw = 'w'
                print("Running SWT calculator with given parameters and hyperparameters...")
                riu_val = swt_calc(varDict[n_list], varDict[d_list], pol_mode, resolution, change = temp_aw,
                                    a_i = first_angle, a_f = end_angle, w_c = fix_angle, w_i = start_wave, 
                                    w_f = end_wave, a_c = fix_angle, verb = 1)
        except:
            print(exception_gen)
    
    ##########################
    
    elif command == 'efield sim':
        try:
            while True:
                print("Simulation of the electric field accross a multilayer when light passes through it.")
                n_list = input("Please type in which variable corresponds to the index list you want to use.\n\n")
                if n_list == "exit":
                    print(back_main)
                    break
                elif n_list not in varDict:
                    try:
                        while n_list not in varDict:
                            print(str(n_list), " not found in stored in variables.")
                            n_list = input("Please input another variable name for the indexes to use.\n\n")
                            if n_list == 'exit':
                                raise exit
                    except exit:
                        print(exception_gen)
                d_list = input("Please type in which variable corresponds to the depth list you want to use.\n\n")
                if d_list == "exit":
                    print(back_main)
                    break
                elif d_list not in varDict:
                    try:
                        while d_list not in varDict:
                            print(str(to_graph), " not found in stored in variables.")
                            d_list = input("Please input another variable for the depth list to use.\n\n")
                            if d_list == 'exit':
                                raise exit
                    except exit:
                        print(exception_gen)
                print("Running simulation with given parameters and hyperparameters......")
                result = multilayer(varDict[n_list], varDict[d_list], mode = pol_mode, step = resolution, 
                                    e_opt = 'y', w_t = fix_wave, ang = fix_angle, limit = efield_limit)
                print("Done!")
                command = input("Do you wish to graph the result? (yes/no)\n\n")
                if command == "yes":
                    label = []
                    results = []
                    d_set = []
                    text = str(fix_angle), "°, ", str(fix_wave), "nm"
                    label.append(text)
                    results.append(result)
                    d_set.append(d_list)
                    graph(results, label, efield = 1, d_set = d_set)
                elif command == "exit":
                    print(back_main)
                    break   
                command = input("Do you wish to save the data from the generated simulation? (yes/no)\n\n")
                if command == "yes":
                    var = input("Please type in to what variable you'd like to save the result.\n\n")
                    if var == "exit":
                        print(back_main)
                        break
                    else:
                        varDict[var] = result.copy()
                        print("Result sucesfully generated and stored in ", var, ".")
                elif command == "exit":
                    print(back_main)
                print("Returning to input for electrical field simulation. Type \"exit\" to return to main.")
        except:
            print(exception_gen)

    ##########################
    
    elif command == 'bloch detector':
        try:
            print("Subroutine to detect if a specific optical multilayer arrangement exhibits Bloch Surface Waves.")
            print("Tip: there is no way to save the given angle/wavelength and minimas within the program, so write them down elsewhere if you wish to use them.")
            while True:
                n_list = input("Please insert the index variable corresponding to the multilayer you wish to test.\n\n")
                if n_list == 'exit':
                    print(back_main)
                    break
                elif n_list == 'help':
                    print(bloch_help)
                    continue
                elif n_list not in varDict:
                        try:
                            while n_list not in varDict:
                                print(str(n_list), " not found in stored in variables.")
                                n_list = input("Please input another variable name for the indexes to use.\n\n")
                                if n_list == 'exit':
                                    raise exit
                        except exit:
                            print(exception_gen)
                d_list = input("Please insert the depth variable corresponding to the multilayer you wish to test.\n\n")
                if d_list == 'exit':
                    print(back_main)
                    break
                elif d_list == 'help':
                    print(bloch_help)
                    continue
                elif d_list not in varDict:
                        try:
                            while d_list not in varDict:
                                print(str(n_list), " not found in stored in variables.")
                                d_list = input("Please input another variable name for the withs to use.\n\n")
                                if d_list == 'exit':
                                    raise exit
                        except exit:
                            print(exception_gen)
                print("Running Bloch Surface Wave detector with the given variables and hyperparameters...")
                if iterate == 'angle':
                    temp_aw = 'a'
                    if crit_ang_set == 'no':
                        first_angle = start_angle
                    elif crit_ang_set == 'yes':
                        first_angle = crit_ang(varDict[n_list])
                elif iterate == 'wavelength':
                    temp_aw = 'w'
                pos, coord, minimum = bloch_wave(varDict[n_list], varDict[d], mode = pol_mode, step = resolution, option_aw = temp_aw, a1 = first_angle,
                            a2 = end_angle, w_t = fix_wave, w1 = start_wave, w2 = end_wave, 
                            ang = fix_angle, roof = 0.90, minimal = min_required, verb = 0)
                del temp_aw
                if pos == False:
                    print("No Bloch Surface Wave detected in the given configuration.")
                    print(back_sub)
                else:
                    print("One or more Bloch Surface Waves detected.")
                    print("Bloch Surface Waves detected at" + iterate + "s: ")
                    for i in pos:
                        print(pos)
                    print("Reflectivity minimas detected: ")
                    for i in minimum:
                        print(minimum)
        except:
            print(exception_gen)
    
    ##########################
    
    elif command == 'multilayer sim dynamic':
        try:
            print("Simulation of light passing through an optical multilayer, with dynamic indexes.")
            while True:
                n_list = [0]
                n_dispatch = input("Please type in which variable corresponds to the index key dispatch list you want to use.\n\n")
                if n_dispatch == "exit":
                    print(back_main)
                    break
                elif n_dispatch == 'help':
                    print(multilayer_sim_dyn_help)
                    continue
                elif n_dispatch not in varDict:
                        try:
                            while n_dispatch not in varDict:
                                print(str(n_list), " not found in stored in variables.")
                                n_list = input("Please input another variable name for the index key dispatch list to use.\n\n")
                                if n_dispatch == 'exit':
                                    raise exit
                        except exit:
                            print(exception_gen)
                d_list = input("Please type in which variable corresponds to the depth list you want to use.\n\n")
                if d_list == "exit":
                    print(back_main)
                    break
                elif d_list == 'help':
                    print(multilayer_sim_dyn_help)
                    continue
                elif d_list not in varDict:
                            try:
                                while d_list not in varDict:
                                    print(str(to_graph), " not found in stored in variables.")
                                    d_list = input("Please input another variable for the depth list to use.\n\n")
                                    if d_list == 'exit':
                                        raise exit
                            except exit:
                                print(exception_gen)
                print("Running simulation with given parameters and hyperparameters......")
                if iterate == "angle":
                    temp_aw = "a"
                elif iterate == "wavelength":
                    temp_aw = "w"
                result = multilayer(n_list, varDict[d_list], mode = pol_mode, step = resolution, 
                        option_aw = temp_aw, a1 = start_angle, a2 = end_angle, w_t = fix_wave,
                        w1 = start_wave, w2 = end_wave, ang = fix_angle, n_wv_dic = dynamic_formulas,
                        nw_list = varDict[n_dispatch])
                del temp_aw
                print("Done!")
                command = input("Do you wish to graph the result? (yes/no)\n\n")
                if command == "yes":
                    label = []
                    results = []
                    if iterate == "angle":
                        text = (str(fix_wave), " nm")
                    elif iterate == "wavelength":
                        text = str(fix_angle, " degrees")
                    label.append(text)
                    results.append(result)
                    graph(results, label)
                elif command == "exit":
                    print(back_main)
                    break   
                command = input("Do you wish to save the data from the generated simulation? (yes/no)\n\n")
                if command == "yes":
                    var = input("Please type in to what variable you'd like to save the result.\n\n")
                    if var == "exit":
                        print(back_main)
                        break
                    else:
                        varDict[var] = copy.copy(result)
                        print("Result sucesfully generated and stored in ", var, ".")
                elif command == "exit":
                    print(back_main)
                print("Returning to input for mutilayer simulation. Type \"exit\" to return to main.")
        except:
            print(exception_gen)

    ##########################
    
    elif command == 'dynamic formulas':
        try:
        #if True:
            print("Subroutine to input dynamic formulas dependent on wavelength for optical multilayer simulations.")
            while True:
                name = input("Please type in the name of the formula you want to input. Type \"exit\" to quit.\n\n")
                if name == 'exit':
                    print(back_main)
                    break
                elif name == 'help':
                    print(dyn_for_help)
                    continue
                formula = input("Please type in the formula you want to input, use \"x\" for the variable.\n\n")
                if formula == 'exit':
                    print(back_main)
                    break
                else:
                    text_to_form = "def " + name + "_form(x):\n    n = " + formula + "\n    return n"
                    text_to_form = str(text_to_form)
                    exec(text_to_form, globals())
                    text_to_def = "dynamic_formulas[name] = " + name + "_form"
                    text_to_def = str(text_to_def)
                    exec(text_to_def, globals())
                    print("Index function ", name, " succesfully created and stored.")
        except:
            print(exception_gen)

    ##########################
    
    elif command == 'dynamic index':
        print("Subroutine to assemble a list of function keys for dynamic indexing.")
        try:
            while True:
                n_keys = []
                name = input("Please insert a variable name for the key list.\n\n")
                if name == 'exit':
                    print(back_main)
                    break
                elif name == 'help':
                    print(dynamic_ind_help)
                    continue
                command = input("Do you wish to use the multilayer autobuilder for assembling the index list? (yes/no)\n\n")
                if command == 'exit':
                    print(back_main)
                    break
                elif command == 'help':
                    print(dynamic_ind_help)
                    continue
                elif command == 'yes':
                    var_list = []
                    period = input("Please enter the periodicity of the multilayer.\n\n")
                    if period == 'exit':
                        print(back_main)
                        break
                    number = input("Please enter how many sets of layer are in the multilayer.\n\n")
                    if number == 'exit':
                        print(back_main)
                        break
                    top = input("Please enter the key name for the index function for the top medium.\n\n")
                    if top == 'exit':
                        print(back_main)
                        break
                    bottom = input("Please enter the key name for the index function for the bottom medium.\n\n")
                    if bottom == 'exit':
                        print(back_main)
                        break
                    n_keys.append(top)
                    for i in range(1, period+1):
                        layer = input("Please enter the key name for the index function for layer ", str(i), ".\n\n")
                        var_list.append(layer)
                    for i in range(0, number):
                        for s in range(0, period):
                            n_keys.append(var_list[s])
                    n_keys.append(bottom)
                    varDict[name] = copy.copy(n_keys)
                    print("Key dispatch list with variable name ", name, " sucesfully created.")
                    print(back_sub)
                elif command == 'no':
                    print("You will be prompted to insert the name of the function to be used for the index of the multilayer, in order from top to bottom.")
                    print("Type \"exit\" to quit.")
                    while True:
                        key = input("Please insert an appropiate key name for an index function.")
                        if key == 'exit':
                            print(back_sub)
                            if len(n_keys) != 0:
                                varDict[name] = copy.copy(n_keys)
                                print("Key dispatch list with variable name ", name, " succesfully created.")
                            break
                        elif key not in dynamic_formulas:
                            print("Error: key not found formula list.")
                        else:
                            n_keys.append(key)
                            
                else:
                    print("Invalid command.")
        except:
            print(exception_gen)
    
    ##########################
    
    elif command == 'graph':
        try:
            while True:
                number_graphs = input("Please input how many datasets you want to graph at once.\n\n")
                if number_graphs == 'exit':
                    print(back_main)
                    break
                elif number_graphs == 'help':
                    print(graph_help)
                    continue
                else:
                    number_graphs = int(number_graphs)
                    label_list = []
                    graph_list = []
                    for i in range(1, number_graphs + 1):
                        to_graph = input("Please input which saved simulation you wish to graph.\n\n")
                        if to_graph == 'exit':
                            print(back_sub)
                            break
                        elif to_graph not in varDict:
                            try:
                                while to_graph not in varDict:
                                    print(str(to_graph), " not found in stored in variables.")
                                    to_graph = input("Please input another variable name for a simulation to graph.\n\n")
                                    if to_graph == 'exit':
                                        raise exit
                            except exit:
                                print(exception_gen)
                        else:
                            graph_list.append(varDict[to_graph])
                        to_label = input("Please input how you want to label this data set on the graph.\n\n")
                        if to_label == 'exit':
                            print(back_sub)
                            break
                        else:
                            label_list.append(to_label)
                    print("Graphing data loaded...running graphing function.")
                    graph(graph_list, label_list)
                    print("Returning to input for \"graph\" subroutine, input \"exit\" to quit it.")
        except:
            print(except_gen)
    
    ##########################
    
    elif command == 'graph efield':
        try:
            while True:
                number_graphs = input("Please input how many datasets you want to graph at once.\n\n")
                if number_graphs == 'exit':
                    print(back_main)
                    break
                elif number_graphs == 'help':
                    print(graph_efield_help)
                    continue
                elif is_float(number_graphs) == False:
                    print(invalid_command)
                    continue
                number_graphs = int(number_graphs)
                label_list = []
                graph_list = []
                depth_list = []
                for i in range(1, number_graphs + 1):
                    to_graph = input("Please input which saved simulation you wish to graph.\n\n")
                    if to_graph == 'exit':
                        print(back_sub)
                        break
                    elif to_graph not in varDict:
                        try:
                            while to_graph not in varDict:
                                print(str(to_graph), " not found in stored in variables.")
                                to_graph = input("Please input another variable name for a simulation to graph.\n\n")
                                if to_graph == 'exit':
                                    raise exit
                        except exit:
                            print(exception_gen)
                    graph_list.append(varDict[to_graph])
                    to_depth = input("Please input which saved depth variable corresponds to this simulation.\n\n")
                    if to_depth == 'exit':
                        print(back_sub)
                        break
                    elif to_depth not in varDict:
                        try:
                            while to_depth not in varDict:
                                print(str(to_depth), " not found in stored in variables.")
                                to_graph = input("Please input another variable name for the widths to be used.\n\n")
                                if to_graph == 'exit':
                                    raise exit
                        except exit:           
                            print(exception_gen)
                    depth_list.append(to_depth)
                    to_label = input("Please input how you want to label this data set on the graph.\n\n")
                    if to_label == 'exit':
                        print(back_sub)
                        break
                    else:
                        label_list.append(to_label)
                print("Graphing data loaded...running graphing function.")
                graph(graph_list, label_list, efield = 1, d_set = depth_list)
                print("Returning to input for \"graph\" subroutine, input \"exit\" to quit it.")
        except:
            print(except_gen)

    ##########################
    
    elif command == 'explore':
        try:
            while True:
                print("Subroutine to explore the parameter space for multilayer designs to maximize sensitivity.")
                print("Warning: RIU and SWT filtering gives unexpected results currently. Will be investigated and fixed ASAP.")
                while True:
                    try:
                        run_name = input("Please input the name under which you want to save the generated multilayer depth arrangements.\n\n")
                        if run_name == 'exit':
                            raise exception_gen
                        elif run_name == 'help':
                            print(explore_help)
                        else:
                            break
                    except:
                        print(exception_gen)
                while True:
                    try:
                        n_list = input("Please input the multilayer index variable for this explore.")
                        if n_list == 'exit':
                            raise exception_gen
                        elif n_list == 'help':
                            continue
                            print(explore_help)
                        elif n_list not in varDict:
                            print("Index variable not found in stored variables.")
                            continue
                        else:
                            break
                    except:
                        print(exception_gen)
                while True:
                    try:
                        layer_exp = input("Please input how much should the regular layers to be expanded upon in simulations. (in meters)\n\n")
                        if layer_exp == 'exit':
                            raise exception_gen
                        elif n_list == 'help':
                            continue
                            print(explore_help)
                        elif is_float(layer_exp) == False:
                            print(invalid_command)
                            continue
                        else:
                            break
                    except:
                        print(exception_gen)
                while True:
                    try:
                        defect_exp = input("Please input how much should the defect layer to be expanded upon in simulations. (in meters)\n\n")
                        if defect_exp == 'exit':
                            raise exception_gen
                        elif n_list == 'help':
                            continue
                            print(explore_help)
                        elif is_float(defect_exp) == False:
                            print(invalid_command)
                            continue
                        else:
                            break
                    except:
                        print(exception_gen)
                while True:
                    try:
                        incr_set = input("Please input at what increment should the code increase the widths of the layers. (in nanometers)\n\n")
                        if incr_set == 'exit':
                            raise exception_gen
                        elif n_list == 'help':
                            continue
                            print(explore_help)
                        elif is_float(incr_set) == False:
                            print(invalid_command)
                            continue
                        else:
                            break
                    except:
                        print(exception_gen)
                while True:
                    try:
                        print("Please input the verbosity you want the function to have" +
                            " (minimal = \"0\", intermediate = \"1\", maximum = \"2\").")
                        verb_set = input()
                        if verb_set == 'exit':
                            raise exception_gen
                        elif verb_set == 'help':
                            print(explore_help)
                            continue
                        elif (verb_set != 0) or (verb_set != 1) or (ver_set != 2):
                            print("Invalid input.")
                            continue
                        else:
                            break
                    except:
                        print(exception_gen)
                print("Running multilayer explore function with given parameters and hyperparameters...")
                results, _ = multilayer_explore(varDict[n_list], pol, steps, change = iterate, 
                                        a_i = start_angle, a_f = end_angle, w_c = fix_wave, w_i = start_wave, w_f = end_wave, 
                                        a_c = fix_angle, def_ext = defect_exp, nm_ext = layer_exp, incr = incr_set,
                                        low = set_low, riu_set = riu_set, riu_cond = riu_cond, swt_set = swt_set, swt_cond = swt_cond, verb = verb_set)
                cntr = 1
                for i in results:
                    varDict[run_name + "_" + str(cntr)] = i
                    cntr += 1
                print(str(cntr) + " multilayer width variables created.")
                print("Created multilayers saved as " + run_name + "_iterationnumber(ex: " + run_name + "_1 for the first multilayer saved).")
                break
        except:
            print(exception_gen)

    ##########################
    
    elif command == 'execute dev testing':
        try:
            print("Development command to execute code lines interactively WARNING, PYTHON \"exec()\" IS HIGHLY UNSTABLE.")
            while True:
                command = input("Input the Python code you want to exec.\n\n")
                if command == 'exit':
                    print("back_main")
                    break
                elif command == 'help':
                    print("There is no rest for the wicked.")
                    continue
                exec(command, globals())
        except:
            print(exception_gen)

    ##########################
    
    else:
        print(invalid_command)
