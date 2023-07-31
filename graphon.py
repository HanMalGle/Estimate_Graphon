from copy import copy
import numpy as np
from utils import matToFct, fctToFct, fctToMat, colorBar
import warnings
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Define Graphon Class
class Graphon:
    
    def __init__(self,fct=None,mat=None,size=501):
        # fct = specific graphon function, mat = approx. graphon function on regular grid, size = fineness of the graphon matrix
        if fct is None:
            if mat is None:
                raise TypeError('no informations about the graphon')
            self.mat = copy(np.asarray(mat))
            self.fct = matToFct(self.mat)
            self.byMat = True
        else:
            self.fct = fctToFct(fct)
            self.mat = fctToMat(fct,size)
            self.byMat = False
            if not mat is None:
                if not np.array_equal(np.round(fctToMat(fct,mat.shape), 5),np.round(mat, 5)):
                    warnings.warn('the partitioning of the graphon in a grid \'mat\' is not exactly according to the graphon function \'fct\' or might be rotated')
                    print('UserWarning: the partitioning of the graphon in a grid \'mat\' is not exactly according to the graphon function \'fct\' or might be rotated')
    def showColored(self, vmin=None, vmax=None, vmin_=0.01, log_scale=False, ticks = [0, 0.25, 0.5, 0.75, 1], showColorBar=True, colorMap = 'plasma_r', fig_ax=None, make_show=True, savefig=False, file_=None):
        if (self.mat.min() < -1e-3) or (self.mat.max() > 1+1e-3):
            warnings.warn('graphon has bad values, correction has been applied -> codomain: [0,1]')
            print('UserWarning: graphon has bad values, correction has been applied -> codomain: [0,1]')
        self_mat = np.minimum(np.maximum(self.mat,0),1)
        if fig_ax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = fig_ax
        if vmin is None:
            vmin = self_mat.min()
        vmin_diff = np.max([vmin_ - vmin, 0])
        if vmax is None:
            vmax = self_mat.max()
        plotGraphon = ax.matshow(self_mat + vmin_diff, cmap=plt.get_cmap(colorMap), interpolation='none', norm=LogNorm(vmin=vmin + vmin_diff, vmax=vmax + vmin_diff)) if log_scale else \
        ax.matshow(self_mat, cmap=plt.get_cmap(colorMap), interpolation='none', vmin=vmin, vmax=vmax)
        plt.xticks(self_mat.shape[1] * np.array(ticks) - 0.5, [(round(round(i,4)) if np.isclose(round(i,4), round(round(i,4))) else round(i,4)).__str__() for i in ticks])
        plt.yticks(self_mat.shape[0] * np.array(ticks) - 0.5, [(round(round(i,4)) if np.isclose(round(i,4), round(round(i,4))) else round(i,4)).__str__() for i in ticks])
        plt.tick_params(bottom=False)
        if showColorBar:
            ticks_CBar = [((10**(np.log10(vmin + vmin_diff) - i * (np.log10(vmin + vmin_diff) - np.log10(vmax + vmin_diff)) / 5)) if log_scale else ((i/5) * (vmax - vmin) + vmin)) for i in range(6)]
            cbar = colorBar(plotGraphon, ticks = ticks_CBar)
            cbar.ax.minorticks_off()
            cbar.ax.set_yticklabels(np.round(np.array(ticks_CBar) - (vmin_diff if log_scale else 0), 4))
        if make_show:
            plt.show()
        if savefig:
            plt.savefig(file_)
            plt.close(plt.gcf())
        else:
            return(eval('plotGraphon' + (', cbar' if showColorBar else '')))
    def showExpDegree(self,size=101,norm=False,fmt='-',title=True,make_show=True,savefig=False,file_=None):
        if self.byMat:
            g_ = self.mat.mean(axis=0)
            us = np.linspace(0,1,self.mat.shape[1])
        else:
            g_ = fctToMat(fct=self.fct,size=(10*size,size)).mean(axis=0)
            us = np.linspace(0,1,size)
        if norm:
            plt.ylim((-1/20,21/20))
        plt.xlim((-1/20,21/20))
        plotDegree = plt.plot(us, g_, fmt)
        if title:
            plt.xlabel('u')
            plt.ylabel('g(u)')
        plt.gca().set_aspect(np.abs(np.diff(plt.gca().get_xlim())/np.diff(plt.gca().get_ylim()))[0])
        if make_show:
            plt.show()
        if savefig:
            plt.savefig(file_)
            plt.close(plt.gcf())
        else:
            return(plotDegree)
#out: Graphon Object
#     fct = graphon function, mat = graphon matrix, byMat = logical whether graphon was specified by function or matrix
#     showColored = plot of the graphon function/matrix, showExpDegree = plot of the expected degree profile

