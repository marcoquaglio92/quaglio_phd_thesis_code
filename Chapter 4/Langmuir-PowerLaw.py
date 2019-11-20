import os

import sys

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.path as path

from scipy.optimize import minimize, basinhopping

from sklearn import svm

import time

#Changes the working directory to the directory of the present file
abspath = os.path.abspath(sys.argv[0])
dname = os.path.dirname(abspath)
os.chdir(dname)


execfile('system_identification.py')


np.random.seed(seed=123456789)



#Gas constant

R=0.0019872036      #kcal/molK

#Normalised length of PFR

length=1.0

#Catalyst weight

W=2.0

#Concentration of Nitrogen as inert carrier - fixed

N2=5.7E-2

##True parameters, obtained from the work of Carotenuto et al. 

true_parameters=np.array([266.8, 5.54, 5E4, 25E3, 1E5, 30E3, 1E4, 25E3, 386.8, 47.7, 8.1])

#Initial concentrations

x0=np.array([0.50,0.0000001,0.0000001,4E-3])

sigma=0.015*np.ones(len(x0))




T_range=(273.15+180.0,273.15+260.0)       #K

P_range=(10.0,30.0)                        #bar

F_Ethanol_range=(0.1,2.5)                  # mol/h

design_space=(F_Ethanol_range, P_range, T_range)

levels=(2,2,2)


#Generation of preliminary design

dataset=full_factorial(langmuir_ethanol, design_space, levels, sigma)


initial_guess=np.array([97.1,36.25,0.089,12.95])

up_bound=initial_guess+0.1*initial_guess

low_bound=np.zeros(len(initial_guess))

measurable=[0,1,2,3]

bnds=tuple(map(tuple, np.c_[low_bound,up_bound]))

#Compute Maximum Likelihood estimate (no outlier detection method is adopted here)

result=minimize(loglikelihood, initial_guess, args=(dataset, sigma, powerlaw_ethanol, measurable,), method='SLSQP', bounds=bnds,options={'disp': True, 'ftol': 1e-15})

print(result.fun)

print(initial_guess, result.x)

estimated_parameters=result.x
    


if result.fun>st.chi2.interval(0.90, sum(len(dataset[i][4]) for i in range(0,len(dataset))), loc=0, scale=1)[1]:
    print ('\nFailed chi-square test: chi2 of the sample ',result.fun, 'higher than chi-square of reference ',st.chi2.interval(0.90, sum(len(dataset[i][4]) for i in range(0,len(dataset))), loc=0, scale=1)[1])
else:
    print ('\nSuccesful chi-square test: chi2 of the sample ',result.fun, 'lower than chi-square of reference ',st.chi2.interval(0.90, sum(len(dataset[i][4]) for i in range(0,len(dataset))), loc=0, scale=1)[1])

# If the chi-square test is not successful then it is justified the employment of data mining

#covariance=-nd.Hessian(loglikelihood)(estimated_parameters, dataset, sigma, powerlaw_ethanol, measurable)

#t-test for the statistical quality of parameters


information=sum(expected_fisher(dataset[i,0],dataset[i,1],length,estimated_parameters,sigma, powerlaw_ethanol, dataset[i,4]) for i in range(0,len(dataset)))


#The statistical quality of the estimates is evaluated
#in the following lines. The array t_test_p contains boolean True-False values
#which indicate if the parameters passed or failed the t-test


covariance=np.linalg.inv(information)

variances=np.diag(covariance)

t_values=estimated_parameters/(st.t.interval(0.95,sum(len(dataset[i,4]) for i in range(0,len(dataset))))[1]*np.sqrt(variances))

t_ref=st.t.interval(0.90,sum(len(dataset[i,4]) for i in range(0,len(dataset))))

t_test_p=t_values>t_ref[1]



#Application of data mining
#for a varying value of the tolerance in the range 6.0, 0.5

tolerance_range=np.linspace(0.5,6.0,24)

#The maximum likelihood estimate is used as initial guess

initial_guess=estimated_parameters

up_bound=initial_guess+2.0*initial_guess

low_bound=np.zeros(len(initial_guess))

measurable=[0,1,2,3]

#A number of arrays is initialised to store the values
#that will be computed in the following for loop 

chi2=np.zeros(len(tolerance_range))

switchers=np.zeros(shape=(len(tolerance_range),len(dataset)))

parameter_history=np.zeros(shape=(len(tolerance_range),len(estimated_parameters)))

t_values_history=np.zeros(shape=(len(tolerance_range),len(estimated_parameters)))

t_test_history=np.zeros(shape=(len(tolerance_range),len(estimated_parameters)))

check=np.zeros(len(dataset))


#At each iteration of the following loop, a different value of the MBDM tolerance is tested

for i,tolerance in reversed(list(enumerate(tolerance_range))):
    
    result=minimize(MBDM, estimated_parameters, args=(dataset, sigma, powerlaw_ethanol, measurable, tolerance,), method='SLSQP', bounds=bnds,options={'disp': True})
    
    #result=minimize(MBDM, result.x, args=(dataset, sigma, powerlaw_ethanol, measurable, tolerance,), method='powell', bounds=bnds,options={'disp': True, 'ftol': 1e-15})
    
    chi2[i],switchers[i]=MBDM_switcher(result.x, dataset, sigma, powerlaw_ethanol, measurable, tolerance)
    
    

    if np.subtract(switchers[i],check).any()!=0.0 or i==len(tolerance_range)-1:
        
        check=switchers[i]
        
        estimated_parameters=result.x
    
        parameter_history[i]=result.x
    
        
        reduced_dataset=dataset[switchers[i]==1]
    
        information=sum(expected_fisher(reduced_dataset[i,0],reduced_dataset[i,1],length,estimated_parameters,sigma, powerlaw_ethanol, reduced_dataset[i,4]) for i in range(0,len(reduced_dataset)))
    
        covariance=np.linalg.inv(information)

        variances=np.diag(covariance)

        t_values_history[i]=estimated_parameters/(st.t.interval(0.95,sum(len(reduced_dataset[j,4]) for j in range(0,len(reduced_dataset))))[1]*np.sqrt(variances))

        t_ref=st.t.interval(0.90,sum(len(reduced_dataset[j,4]) for j in range(0,len(reduced_dataset))))

        t_test_history[i]=t_values_history[i]>t_ref[1]
    
        print(tolerance, chi2[i], switchers[i], t_values_history[i], t_ref, t_test_history[i])
    
    if chi2[i]<st.chi2.interval(0.90, sum(switchers[i])*4, loc=0, scale=1)[1]:
        
        compatible_set=dataset[switchers[i]==1.0]
        
        incompatible_set=dataset[switchers[i]==0.0]
        break
        
#Construction of training set for SVM
#Assuming tolerance equal 2.0, this tolerance corresponds to index 9
#in the switchers and chi2 arrays

#In the following, the relevant quantities, switchers, chi2 (sum of squared residuals),
#t-values, parameter values, chi2 reference, t-values reference
#are stored in the respective lists with _history label.

#Such lists will be populated adding the relevant value at each iteration of the online MBDM campaign

switchers=switchers[9]

chi2=chi2[9]

switchers_history=list()

switchers_history.append(switchers)

t_values_history=list()

t_values_history.append(t_test(result.x, np.linalg.inv(information), 0.95, len(measurable)*sum(switchers))[0])

parameter_history=list()

parameter_history.append(result.x)

chi2_history=list()

chi2_history.append(chi2)

chi2_ref_history=list()

chi2_ref_history.append(st.chi2.interval(0.90, sum(switchers)*4, loc=0, scale=1)[1])

t_ref_history=list()

t_ref_history.append(t_test(result.x, np.linalg.inv(information), 0.95, len(measurable)*sum(switchers))[1])

#np.random.seed(seed=1234567)

#Compute and export the reliability map
#assuming a decay length of 0.5

gamma=0.5

clf=reliability_map(dataset, switchers, gamma)

export_reliability('reliability_map_'+str(0)+'.xlsx')

tol=2.0

max_iteration=15

iteration=1

#significance represents a boolean variable that turns to True when the 
#statistical significance of all the model parameters is satisfactory

significance=False

#Boundaries for experimental design

up_bound_design=np.array([2.5,30.0,273.15+260.0])

low_bound_design=np.array([0.1,10.0,273.15+180.0])

up_bound_design=np.array([1.0,1.0,1.0])

low_bound_design=np.array([0.0,0.0,0.0])

dsgnbnds=map(tuple, np.c_[low_bound_design,up_bound_design])

#The reliability map is imposed as constraint for the experimental design

cons=[{'type':'ineq', 'fun':constraints}]

#Arrays are created also to store the computational times for each algorithm iteration

computational_times_design=np.zeros(shape=(max_iteration, 3))

computational_times_MBDM=np.zeros(max_iteration)


while iteration<=max_iteration and significance==False:
    
    #The number of designed experiments is chosen iteratively in the range 1-3
    #based on the number of predicted experiments required to achieve the desired statistical
    #quality of the parameter estimates
    
    print '\nIteration ', iteration
    
    print 'Evaluation of required number of constrained experiments '
    
    no_experiments_to_design=1
    
    expected_statistics=False
    
    while expected_statistics==False and no_experiments_to_design<=3:
        
        print 'Designed experiments: ', no_experiments_to_design
        
        message=''
        
        while message=='' or message!='Optimization terminated successfully.':
            #The following command generates a random initial guess in the unit hypercube without checking if the initial guess respects the constraints
            #design_vector=np.random.uniform(size=3*no_experiments_to_design)
            #The following command generates a random initial guess within the domain of model reliability
            design_vector=initial_guess_design(no_experiments_to_design)
            try:
                design_tick=time.time()
                design=minimize(exp_design, design_vector, args=(powerlaw_ethanol, estimated_parameters, information, sigma, no_experiments_to_design, 'D', 'objective', ), method='SLSQP', bounds=(map(tuple, np.tile(dsgnbnds,(no_experiments_to_design,1)))), constraints=cons,  options={'disp': True, 'ftol': 1e-20})
                
                #Uncomment the following line to remove the reliability map as a constraint
                #design=minimize(exp_design, design_vector, args=(powerlaw_ethanol, estimated_parameters, information, sigma, no_experiments_to_design, 'D', 'objective', ), method='SLSQP', bounds=(map(tuple, np.tile(dsgnbnds,(no_experiments_to_design,1)))), options={'disp': True, 'ftol': 1e-20})
                computational_times_design[iteration-1, no_experiments_to_design-1]=time.time()-design_tick
                message=design.message    
            except ValueError:
                print 'Error encountered - An additional design is attempted with different initial guesses'
            
        #Computes the predicted covariance of the estimates to decide whether the designed experiments are
        #sufficient to achieve the desired parameter precision
        predicted_covariance=exp_design(design.x, powerlaw_ethanol, estimated_parameters, information, sigma, no_experiments_to_design, 'D', 'covariance')
        
        print t_test(estimated_parameters, predicted_covariance, 0.95, len(measurable)*(sum(switchers)+no_experiments_to_design))
        
        no_experiments_to_design=no_experiments_to_design+1
        
        expected_statistics=t_test(estimated_parameters, predicted_covariance, 0.95, len(measurable)*(sum(switchers)+no_experiments_to_design))[2]

    #Ranking of the designed experiments based on the trace of the expected Fisher information matrix
    
    conditions=np.add(np.transpose(design_space)[0],np.multiply(design.x.reshape(len(design.x)/3,3),np.subtract(np.transpose(design_space)[1],np.transpose(design_space)[0])))

    information_design=predicted_information(conditions, estimated_parameters, sigma, powerlaw_ethanol)
    
    conditions=conditions[np.argsort(information_design)][::-1]

    ranked_designed_experiments=design.x.reshape(len(design.x)/3,3)[np.argsort(information_design)][::-1]
    
    print constraints(ranked_designed_experiments[0])
    
    #additional_best_design=ranked_designed_experiments[1:]  #The additional best design is used at the following iteration as initial guess

    #Execution of best experiment
    
    print '\nThe most informative designed experiment is executed'
    
    dataset=np.append(dataset,experiment([conditions[0,0],0.000001,0.000001,N2],[conditions[0,1],conditions[0,2]],[length],sigma,[2.0,N2],langmuir_ethanol,[0,1,2,3]),axis=0)

    #Evaluates the content in information for the new experiments and sorts the ne design vector in decreasing order of information (trace of expected Fisher)


    #fitting of after executing the most informative experiment:
    
    print '\nThe new dataset is fitted with MBDM with tolerance ', tol
    guess=estimated_parameters
    MBDM_tick=time.time()
    
    #The MBDM problem is solved for progressively smaller tolerance values to identify a robust initial guess
    
    for tolerance in np.linspace(tol,6.0,8)[::-1]:
        result=minimize(MBDM, guess, args=(dataset, sigma, powerlaw_ethanol, measurable, tolerance,), method='SLSQP', bounds=bnds, options={'disp': False})
        guess=result.x
    #result=minimize(MBDM, result.x, args=(dataset, sigma, powerlaw_ethanol, measurable, tolerance,), method='powell', bounds=bnds,options={'disp': True, 'ftol': 1e-15})
    
    chi2,switchers=MBDM_switcher(result.x, dataset, sigma, powerlaw_ethanol, measurable, tol)
    computational_times_MBDM[iteration-1]=time.time()-MBDM_tick
    switchers_history.append(switchers)
    parameter_history.append(result.x)
    chi2_history.append(chi2)
    
    estimated_parameters=result.x
    reduced_dataset=dataset[switchers==1]
    
    information=sum(expected_fisher(reduced_dataset[i,0],reduced_dataset[i,1],length,estimated_parameters,sigma, powerlaw_ethanol, reduced_dataset[i,4]) for i in range(0,len(reduced_dataset)))
    
    print 'Iteration ', iteration
    print 'chi2 ', chi2
    print 't-test ', t_test(result.x, np.linalg.inv(information), 0.95, len(measurable)*sum(switchers))
    print 'Number of performed experiments ', len(dataset)
    t_values_history.append(t_test(result.x, np.linalg.inv(information), 0.95, len(measurable)*sum(switchers))[0])
    t_ref_history.append(t_test(result.x, np.linalg.inv(information), 0.95, len(measurable)*sum(switchers))[1])
    chi2_ref_history.append(st.chi2.interval(0.90, sum(switchers)*4, loc=0, scale=1)[1])

    print 'Update of reliability map'
    
    clf=reliability_map(dataset, switchers, gamma)
    
    print 'Exporting reliability map to Excel'
    
    export_reliability('reliability_map_'+str(iteration)+'.xlsx')
    
    significance=t_test(result.x, np.linalg.inv(information), 0.95, len(measurable)*sum(switchers))[2]
    
    
    
    iteration=iteration+1
    

print 'Model identified'



#conditions=np.array([[a[0][0],a[1][0],a[1][1]] for a in dataset])

#conditions=np.divide(np.subtract(conditions,np.transpose(design_space)[0]),np.subtract(np.transpose(design_space)[1],np.transpose(design_space)[0]))


