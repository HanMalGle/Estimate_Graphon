import numpy as np
from data import GraphFromData
from copy import copy
import warnings 
from estimate import Estimator, TuneLambdaSplineRegAIC
from sampling import Sample
from iteration import iterateEM
from utils import showTraject
import matplotlib.pyplot as plt
import pickle
import os

nKnots = 20  # number of inner knots for the B-spline basis
n_steps = 50  # steps of Gibbs iterations
proposal='logit_norm'  # type of proposal to use for the sampling (options: 'uniform')
ks_rel = np.array([0,0.25,0.75,1])  # relative k's for which the posterior is calculated
use_origFct=True  # logical whether to use the graphon function itself or a discrete approx
averageType='mean'  # specify the kind of posterior average
use_stdVals=True  # logical whether to use standardized Us (-> equidistant)
sigma_prop = 2  # variance of sampling step (-> proposal distribution, only used if proposal == 'logit_norm')


## parameters for the EM algorithm
n_iter = 25  # number of EM interations
rep_start = 1  # start value for number of repetitions in the Gibbs sampling step
rep_end = 25  # end value for number of repetitions in the Gibbs sampling step
it_rep_grow = 5  # iteration from which rep starts to grow
lambda_start = 50  # start value for the penalization parameter
lambda_skip1 = 10  # iterations to skip before optimizing lambda
lambda_lim1 = (3) + lambda_skip1  # (.) = optimized lambdas not to use for the mean penalization parameter
lambda_skip2 = (2) + lambda_lim1  # (.) = iterations to skip before optimizing lambda
lambda_lim2 = (3) + lambda_skip2  # (.) = optimized lambdas to use for the mean penalization parameter
lambda_last_m = 3  # last m iterations at which lambda is optimized again
if np.any([lambda_lim2 >= (n_iter - lambda_last_m), lambda_skip1 <= it_rep_grow]):
    warnings.warn('specification of iterations for estimating lambda should be reconsidered')
    print('UserWarning: specification of iterations for estimating lambda should be reconsidered')

## specify figure sizes
figsize1 = (9, 4)
figsize2 = (9, 5)


## parameters for calculating the illustrated posterior distribution
rep_forPost = 25  # number of repetitions/sequences of the Gibbs sampling for calculating the posterior distribution
useAllGibbs = True  # logical whether to use all Gibbs repetitions for calculating the posterior or simply the mean
distrN_fine = 1000  # fineness of the posterior distribution -> number of evaluation points
data_name = ... # data_ => data_name
data_path = ... # dir_ => data_path
simulate = ... # True or False 확인필요
useIndividGraphs = ...
estMethod = ...
useIndividRandInit = ...
start, stop = ... #start_ , stop_ => start, stop
if_Nth = ...
directory_path = ... # directory_ => directory_path
make_show = False
savefig = True # save or not
plotAll = True  # logical whether to plot auxiliary graphics too
trueInit = False  # logical whether to start with true model (true ordering + true graphon, dominates 'estMethod', only used if simulate == True)
initGraphonEst = False  # logical whether to make an initial estimate of the graphon
## parameters for B-spline regression
k = 1  # order of B-splines (only 0 and 1 are implemented)
nKnots = 20  # number of inner knots for the B-spline basis
canonical = False  # logical whether a canonical representation should be fitted
## parameter for observing convergence
n_eval = 3  # number of evaluation points for the trajectory for observing convergence -> equidistant positions
initCanonical = False  # logical whether to start with canoncial estimation (only used if initGraphonEst == True)
initPostDistr = False  # logical whether to calculate the initial posterior distribution (graphon specification necessary)
log_scale = False if simulate else True  # logical whether to use log_scale for graphon plot
rep_forPost = 25  # number of repetitions/sequences of the Gibbs sampling for calculating the posterior distribution
useSameGraph = np.any([simulate and (not useIndividGraphs), not simulate])
useIndividRandInit = (estMethod=='random') and useIndividRandInit  # useIndividRandInit if (estMethod=='random') else False



result_list = []
for glob_ind in range(start,stop+1):
    #glob_ind = start_

    seed_ = glob_ind
    np.random.seed(seed_)
    
    # nTry => Nth  
    Nth = glob_ind.__str__() + '_'  # specify an identification for the run
    dirExt = directory_path + (Nth if if_Nth else '')

    ### Define graph


    if glob_ind == start:
        graph0 = GraphFromData(data_=data_name, dir_=data_path, estMethod=estMethod)
        graph0.sort(Us_type = 'est')
        N=graph0.N
        
    if glob_ind == start:
        print('Average degree in the network:', graph0.averDeg)  # average number of links per node
        print('Overall density:', graph0.density)  # graph density
        
    if useSameGraph:
        if useIndividRandInit:
            if glob_ind != start:
                graph0.update(Us_est=np.random.permutation(np.linspace(0, 1, N + 2)[1:-1]))
        else:
            if glob_ind == start:
                Us_est_unique = copy(graph0.Us_est)
            else:
                graph0.update(Us_est=Us_est_unique)


    # plot adjacency matrix based on initial ordering
    graph0.showAdjMat(make_show=make_show, savefig=savefig, file_=dirExt + 'adjMat_0.png')

    # plot network with initial ordering
    graph0.showNet(make_show=make_show, savefig=savefig, file_=dirExt + 'network_0.png')


    ### Initial fit of graphon + initial posterior distribution of U_k

    if trueInit:
        warnings.warn('real data example is considered, ground truth is unknown')
        print('UserWarning: real data example is considered, ground truth is unknown')

    if initGraphonEst:
        estGraphonData0 = Estimator(sortG=graph0)
        lambda_ = TuneLambdaSplineRegAIC(estimator=estGraphonData0, lambdaMin=0, lambdaMax=1000, paraDict={'k': k, 'nKnots': nKnots, 'canonical': canonical, 'Us_mult': None})
        estGraphon0 = estGraphonData0.GraphonEstBySpline(k=k, nKnots=nKnots, canonical=initCanonical, lambda_=lambda_, Us_mult=None, returnAIC=False)
        trajMat = estGraphon0.fct(np.arange(1, n_eval + 1) / (n_eval + 1), np.arange(1, n_eval + 1) / (n_eval + 1)).reshape(1, n_eval, n_eval)
        # plot initial graphon estimate
        estGraphon0.showColored(log_scale=log_scale, make_show=make_show, savefig=savefig, file_=dirExt + 'graphon_est_1.png')
        
    else:
        estGraphon0 = None
        trajMat = None


    if trueInit:
        warnings.warn('real data example is considered, ground truth is unknown')
        print('UserWarning: real data example is considered, ground truth is unknown')

    seed2_ = glob_ind + 1
    np.random.seed(seed2_)

    if ks_abs is None:
        ks_abs = np.unique(np.minimum(np.maximum(np.round(ks_rel * N).astype('int') - 1, 0), N - 1))  # absolute k's for which the posterior is calculated
    if initPostDistr:
        if rep_forPost == 0:
            warnings.warn('number of repetitions for calculating the posterior distribution is 0, no calculation is carried out')
            print('UserWarning: number of repetitions for calculating the posterior distribution is 0, no calculation is carried out')
        else:
            # apply Gibbs sampling to the initial U's given the graphon estimate based on the initial ordering
            sample0=Sample(sortG=graph0,graphon=estGraphon0,use_origFct=use_origFct)
            sample0.gibbs(steps=n_steps,proposal=proposal,rep=rep_forPost,sigma_prop=sigma_prop, returnAllGibbs=False, averageType=averageType, updateGraph=False, use_stdVals=None, printWarn=True)
            # calculate and plot the posterior distribution of U_k based on the initial graphon estimate, with k corresponding to the initial ordering
            sample0.showPostDistr(ks_lab=None, ks_abs=ks_abs, Us_type_label='est', Us_type_add='est', distrN_fine=distrN_fine, useAllGibbs=True, EMstep_sign='(1)', figsize=figsize2, mn_=None, useTightLayout=True, w_pad=2, h_pad = 1.5, make_show=make_show, savefig=savefig, file_=dirExt + 'postDistr_0.png')
            sample0.updateGraph(use_stdVals=use_stdVals)
            # plot adjacency matrix based on initial ordering
            graph0.showAdjMat(make_show=make_show, savefig=savefig, file_=dirExt + 'adjMat_1.png')
            # plot network with initial ordering
            graph0.showNet(make_show=make_show, savefig=savefig, file_=dirExt + 'network_1.png')
            if simulate:
                # plot differences to real U's
                graph0.showDiff(Us_type='est', EMstep_sign='(1)', make_show=make_show, savefig=savefig, file_=dirExt + 'Us_diffReal_1.png')



    ### Sample U's and fit graphon again and again

    EM_obj = iterateEM(sortG=graph0,
                       k=k, nKnots=nKnots, canonical=canonical,
                       n_steps=n_steps, proposal=proposal, sigma_prop=sigma_prop, use_origFct=use_origFct, averageType=averageType, use_stdVals=use_stdVals,
                       n_iter=n_iter, rep_start=rep_start, rep_end=rep_end, it_rep_grow=it_rep_grow, rep_forPost=rep_forPost,
                       lambda_start=lambda_start, lambda_skip1=lambda_skip1, lambda_lim1=lambda_lim1, lambda_skip2=lambda_skip2, lambda_lim2=lambda_lim2, lambda_last_m=lambda_last_m,
                       n_eval=n_eval, trajMat=trajMat,
                       startWithEst=(not initGraphonEst) or (initGraphonEst and initPostDistr), estGraphon=estGraphon0,
                       endWithSamp=True, raiseLabNb=initGraphonEst and initPostDistr,
                       returnLambList=True, returnGraphonList=False, returnSampList=False, returnAllGibbs=False,
                       makePlots=plotAll, make_show=make_show, savefig=savefig, simulate=simulate, log_scale=log_scale, dir_=dirExt)



    # plot adjacency matrix based on final ordering
    EM_obj.sortG.showAdjMat(make_show=make_show, savefig=savefig, file_=dirExt + 'adjMat_EM.png')

    # plot network with final ordering
    EM_obj.sortG.showNet(make_show=make_show, savefig=savefig, file_=dirExt + 'network_EM.png')

    # plot trajectory of graphon estimation sequence for specific positions u and v
    showTraject(trajMat=EM_obj.trajMat, make_show=make_show, savefig=savefig, file_=dirExt + 'trajectory_graphonSeq.png')

    # plot final graphon estimate
    EM_obj.estGraphon.showColored(log_scale=log_scale, make_show=make_show, savefig=savefig, file_=dirExt + 'graphon_est_EM.png')


    if plotAll:
        # plot observed vs expected degree profile based on EM ordering
        EM_obj.sortG.showObsDegree(absValues=False, norm=False, fmt = 'C1o', title=False, make_show=make_show, savefig=False)
        EM_obj.estGraphon.showExpDegree(norm=False, fmt = 'C0--', title=None, make_show=make_show, savefig=False)
        plt.xlabel('(i) $u$   /   (ii) $\hat{u}_i^{\;EM}$')
        plt.ylabel('(i) $\hat{g}^{\;EM}(u)$   /   (ii) $degree(i) \;/\; (N-1)$')
        plt.tight_layout()
        if make_show:
            plt.show()
        plt.savefig(dirExt + 'obsVsEM_expDegree.png')
        plt.close('all')

    if rep_forPost != 0:
        # calculate and plot the posterior distribution of U_k based on the final graphon estimate, with k corresponding to the final ordering
        EM_obj.sample.showPostDistr(ks_lab=None, ks_abs=ks_abs, Us_type_label='est', Us_type_add='est', distrN_fine=distrN_fine, useAllGibbs=True, EMstep_sign='EM', figsize=figsize2, mn_=None, useTightLayout=True, w_pad=2, h_pad = 1.5, make_show=make_show, savefig=savefig, file_=dirExt + 'postDistr_EM.png')


    result_ = np.array([Nth, EM_obj.sortG.logLik(graphon = EM_obj.estGraphon), EM_obj.AIC, EM_obj.lambda_])
    print(result_)
    result_list.append(result_)

    graph_simple = {'A': EM_obj.sortG.A, 'labels': EM_obj.sortG.labels_(),
                    'Us_real': EM_obj.sortG.Us_('real'), 'Us_est': EM_obj.sortG.Us_('est')}
    graphon_simple = {'mat': EM_obj.estGraphon.mat, 'nKnots': EM_obj.estGraphon.nKnots,
                      't': EM_obj.estGraphon.t, 'theta': EM_obj.estGraphon.theta,
                      'order': EM_obj.estGraphon.order}
    if rep_forPost != 0:
        sample_simple = {'U_MCMC': EM_obj.sample.U_MCMC, 'U_MCMC_std': EM_obj.sample.U_MCMC_std,
                         'U_MCMC_all': EM_obj.sample.U_MCMC_all, 'acceptRate': EM_obj.sample.acceptRate,
                         'Us_new': EM_obj.sample.Us_new, 'Us_new_std': EM_obj.sample.Us_new_std}
    else:
        sample_simple = None

    with open(dirExt + 'final_result.pkl', 'wb') as output:
        pickle.dump(result_, output, protocol=3)
        pickle.dump(graph_simple, output, protocol=3)
        pickle.dump(graphon_simple, output, protocol=3)
        pickle.dump(sample_simple, output, protocol=3)


    # add parameter settings to a csv file
    fname = directory_path + '_register.csv'
    if not os.path.isfile(fname):
        with open(fname, 'a') as fd:
            fd.write('nTry; logLik; AIC; seed; seed2; estMethod; initGraphonEst; initCanonical; initPostDistr; trueInit; N; k; nKnots; canonical; n_steps; sigma_prop; averageType; use_stdVals; rep_forPost; n_iter; rep_start; rep_end; it_rep_grow; lambda_start; lambda_skip1; lambda_lim1; lambda_skip2; lambda_lim2; lambda_last_m; lambda_; \n')
    with open(fname, 'a') as fd:
        fd.write(Nth + ';' + result_[1] + ';' + result_[2] + ';' + seed_.__str__() + ';' + seed2_.__str__() + '; ' + estMethod.__str__() + '; ' + initGraphonEst.__str__() + '; ' + initCanonical.__str__() + '; ' + initPostDistr.__str__() + '; ' + trueInit.__str__() + '; ' + N.__str__() + '; ' + k.__str__() + '; ' + nKnots.__str__() + '; ' + canonical.__str__() + '; ' + n_steps.__str__() + '; ' + sigma_prop.__str__() + '; ' + averageType + '; ' + use_stdVals.__str__() + '; ' +
                 rep_forPost.__str__() + '; ' + n_iter.__str__() + '; ' + rep_start.__str__() + '; ' + rep_end.__str__() + '; ' + it_rep_grow.__str__() + '; ' + lambda_start.__str__() + '; ' + lambda_skip1.__str__() + '; ' + lambda_lim1.__str__() + '; ' + lambda_skip2.__str__() + '; ' + lambda_lim2.__str__() + '; ' + lambda_last_m.__str__() + '; ' + np.round(EM_obj.lambda_, 4).__str__() + '; \n')


    print('\n\n\nGlobal repetition complete:    ' + glob_ind.__str__() + '\n\n\n\n\n\n\n')
