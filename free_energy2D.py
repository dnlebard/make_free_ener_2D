#!/usr/bin/env ipython

import sys
import os
import numpy  as np
import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter, MaxNLocator
from numpy import linspace
from matplotlib import cm
import argparse



#################################################################
def plot_free_ener(input2d,output2d,xmin=None,xmax=None,ymin=None,ymax=None):
    """ Plot the 2-D free energies (in kcal/mol) from the 2-D input file """

    input_file = os.path.abspath(input2d)

    # Get output directory path
    if not os.path.dirname(output2d):
        output_dir = os.path.abspath(".")
    else:
        output_dir = os.path.abspath(os.path.dirname(output2d))

    # go to output_dir
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    os.chdir(output_dir)

    print "Now working in directory: %s" % output_dir

    print 'NOW USING NUMPY VERSION %s ' % np.version.version

    print 'NOW WORKING WITH FILE NAMED %s ' % input_file

    x,y = np.loadtxt(input_file, unpack=True)    #np.fromfile(input_file, dtype=np.float32)


    if  xmin is None:
        xmin = min(x)
    if  xmax is None:
        xmax = max(x)
    if  ymin is None:
        ymin = min(y)
    if  ymax is None:
        ymax = max(y)

    print 'Using xmin=%d, xmax=%d, ymin=%d, ymax=%d' % (xmin,xmax,ymin,ymax)

    # Set up default x and y limits
    xlims = [xmin,xmax]
    ylims = [ymin,ymax]


    nxbins = 50
    nybins = 50
    nbins = 100

    xbins = linspace(start = xmin, stop = xmax, num = nxbins)
    ybins = linspace(start = ymin, stop = ymax, num = nybins)

    xcenter = (xbins[0:-1]+xbins[1:])/2.0
    ycenter = (ybins[0:-1]+ybins[1:])/2.0
    #aspectratio = 1.0*(xmax - 0)/(1.0*ymax - 0)

    #H, xedges,yedges = np.histogram2d(y,x,bins=(ybins,xbins),normed=True)
    # # --> From example: H, xedges, yedges = np.histogram2d(x,y,bins=nbins)
    #H, xedges,yedges = np.histogram2d(x,y,bins=(ybins,xbins),normed=True)
    H, xedges,yedges = np.histogram2d(y,x,bins=(ybins,xbins),normed=True)
    X = xcenter
    Y = ycenter
    Z = H

    # H needs to be rotated and flipped
    #H = np.rot90(H)
    #H = np.flipud(H)

    # Set up your x and y labels
    xlabel = '$\mathrm{Rg\\ (nm)}$'
    ylabel = '$\mathrm{Asphericity\\ (nm^2)}$'

    #free_ener = 0.593*(np.log(H))
    H = -0.593*(np.log(H + 0.0000000001))

    maxH = np.amax(H)

    H = H - maxH

    eval_val = -0.593*(np.log(0.0000000001)) - maxH

    # Mask zeros
    #Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero
    Hmasked = np.ma.masked_where(H==eval_val,H) # Mask pixels with a value of zero

    #free_ener_masked = np.ma.masked_where(free_ener==0,free_ener) # Mask pixels with a value of zero


    # Plot 2D histogram using pcolor
    fig2 = plt.figure()
    #plt.pcolormesh(xedges,yedges,Hmasked)

    #plt.pcolormesh(yedges,xedges,Hmasked,cmap=cm.jet_r) # cmap=jet_r is the reverse rainbow color setting --- it works, too
    plt.pcolormesh(yedges,xedges,Hmasked,cmap=cm.gnuplot_r)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Free energy (kcal/mol)')

    cbar.ax.invert_yaxis() #make the ylabel inverted (small negative numbers on bottom, large on top)

    # Define the locations for the axes
    #left, width = 0.12, 0.55
    #bottom, height = 0.12, 0.55
    #bottom_h = left_h = left+width+0.02
    #
    #rect_temperature = [left, bottom, width, height] # dimensions of temp plot
    #rect_histx = [left, bottom_h, width, 0.25] # dimensions of x-histogram
    #rect_histy = [left_h, bottom, 0.25, height] # dimensions of y-histogram

    # Set up the size of the figure
    #fig = plt.figure(1, figsize=(9.5,9))
    #
    ## Make the plots
    #axTemperature = plt.axes(rect_temperature) # temperature plot
    #
    #cax = (axTemperature.imshow(H, extent=[xmin,xmax,ymin,ymax],interpolation='nearest', origin='lower',aspect=aspectratio))
    #
    ##Plot the axes labels
    #axTemperature.set_xlabel(xlabel,fontsize=25)
    #axTemperature.set_ylabel(ylabel,fontsize=25)
    #
    #
    ##Plot the axes labels
    #axTemperature.set_xlabel(xlabel,fontsize=25)
    #axTemperature.set_ylabel(ylabel,fontsize=25)
    #
    ##Make the tickmarks pretty
    #ticklabels = axTemperature.get_xticklabels()
    #for label in ticklabels:
    #    label.set_fontsize(18)
    #    label.set_family('serif')
    #
    #ticklabels = axTemperature.get_yticklabels()
    #for label in ticklabels:
    #    label.set_fontsize(18)
    #    label.set_family('serif')
    #
    ##Set up the plot limits
    #axTemperature.set_xlim(xlims)
    #axTemperature.set_ylim(ylims)




    #Show the plot
    plt.draw()
    plt.show()


    print 'LIMITS ON X(Rg): %d to %d ; AND FOR Y(asp): %d to %d' % (min(x),max(x),min(y),max(y))

    #x = np.arange(0, 5, 0.1);
    #y = np.sin(x)
    #plt.plot(x, y)
    #plt.show()
    #nfo_file = open(nfo_file_name,"w")
    #nfo_file.write('1\n')
    #nfo_file.write('1\n')
    #nfo_file.write('1\n')
    #nfo_file.write('1\n')
    #    nfo_file.close()


#################################################################

################# ARGUMENT PARSING ##############################

EXPECTED_ARGS = 2
TOT_ARGS = 6
num_args = len(sys.argv) - 1 # The first argument (sys.argv[0]) is the script name

exe_name = os.path.basename(sys.argv[0])
if  num_args != EXPECTED_ARGS and num_args != TOT_ARGS:
    print '\nUsage: %s 2D_input_file 2D_output_file (in kcal/mol) <xmin> <xmax> <NEGATIVE_ymin> <ymax>' % exe_name
    print '\nExample 1: %s input.dat free-energy/free_energy.dat              ... OR ...' % exe_name
    print 'Example 2: %s input.dat free-energy/free_energy.dat 0 10 -10 10 \n' % exe_name

    sys.exit()

input2d   = sys.argv[1]
output2d  = sys.argv[2]
#################################################################

# Now Call the plotting routine with the appropriate arguments
if num_args == TOT_ARGS:
    xmin = float(sys.argv[3])
    xmax = float(sys.argv[4])
    ymin = -1.0*float(sys.argv[5])
    ymax = float(sys.argv[6])
    plot_free_ener(input2d,output2d,xmin,xmax,ymin,ymax)
else:
    plot_free_ener(input2d,output2d)
