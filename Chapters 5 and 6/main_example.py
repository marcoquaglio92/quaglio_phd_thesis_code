import os
import sys
import itertools

#Changes the working directory to the directory of the file 'main_example.py'
abspath = os.path.abspath(sys.argv[0])
dname = os.path.dirname(abspath)
os.chdir(dname)

'''
The system_identification.py package includes a number of function to:
- Simulate experiments (adding measurement noise);
- Fit model parameters to experimental data;
- Evaluate the statistical quality of the parameter estimates;
- Evaluate the model Goodness-of-fit;
- Diagnose model misspecification through the computation of the Model Modification Index and the Effect Relevance Index;

The package supports the identification of parametric models in the form of Ordinary Differential Equations
'''

#Import all the functions from the system_identification.py Python package
from system_identification import *

np.random.seed(0)

'''
The aim in this script is to demonstrate the use of the system_identification package through the following steps:

-Generation of an in-silico dataset through the integration of the model 'cantois' (a dynamic model of baker's yeast growth in a fed-batch reactor)
-The generated dataset is fitted to estimate the kinetic parameters in a misspecified model, namely the model 'monod'.
-The covariance of the parameter estimates is computed and results of statistical tests on parameter precision and goodness-of-fit are returned.
-The misspecified model 'monod' is falsified by the goodness-of-fit test after the fitting.
-The model modification indexes (MMI) are computed for all the model parameters to detect which model parameters are most likely to hide state-dependencies.
-A set of effect relevance indexes (ERI) are computed for the parameter with the highest MMI to detect which candidate effects are most relevant
 for the evolution of the critical parameter into a state-dependent function.
'''


'''
The equations describing the physical system are provided as a string which contains python code. 
This string represents a numpy array which contains the expression for the first order derivatives of the states

The model cantois involves:

-2 ordinary differential equations;
-2 state variables: x[0] (biomass concentration, measurable); x[1] (substrate concentration, measurable); 
-2 time-invariant input variables: u[1] (dilution factor); u[2] (substrate concentration in the feed);
-4 non-measurable parameters theta[0] to theta[3];
'''

cantois='np.array([((theta[0]*x[1]/(theta[1]*x[0]+x[1]))-u[0]-theta[3])*x[0],-((theta[0]*x[1]/(theta[1]*x[0]+x[1]))*x[0])/theta[2]+u[0]*(u[1]-x[1])])' 


#A set of true parameter values for the parameters involved in 'cantois' is assumed to simulate the kinetic experiments

true_parameters=np.array([0.310, 0.180, 0.550, 0.050])    

#The measurement noise associated with a sample of the state variables x=[x[0], x[1]]
#is quantified as Gaussian, uncorrelated noise with the following standard deviations for states x[0] and x[1] respectively

sigma=np.array([0.05, 0.05])

#The array 'measurable' is an array of indexes and it is used to indicate which state variables in the array x
#are effectively measured in the system. In this case study, since both x[0] and x[1] can be measured, the 
#array measurable is np.array([0,1])

measurable=np.array([0,1])

'''
A full-factorial Experimental design is constructed with:
    
-1 level for the initial biomass concentration;
-1 level for the initial substrate concentration;
-2 levels for the dilution factor;
-2 levels for the substrate concentration in the feed;
-7 levels for the sampling times
'''

initial_x1 = np.array([1.0])

initial_x2 = np.array([0.01]) 

dilution_factor=np.array([0.05, 0.20])

substrate_concentration_in_feed=np.array([5.0, 35.0])

sampling_times=np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0])

experimental_conditions=np.array(list(itertools.product(initial_x1, initial_x2, dilution_factor,substrate_concentration_in_feed,sampling_times)))

#generation of the dataset

#initialise dataset with 2 states, 2 experimental conditions, 1 sampling time, 2 measured states
dataset=np.array([[[None, None], [None, None], None, [None, None], [None, None]]]) #This is the format for the dataset that is accepted by the system_identification.py package

for conditions in experimental_conditions:
    
    dataset=np.append(dataset, np.array(experiment([conditions[0],conditions[1]],[conditions[2],conditions[3]], [conditions[4]], sigma, true_parameters, cantois, measurable )) ,axis=0)
        
dataset=dataset[1:]    


'''
It is assumed that the scientist does not know the exact form of the model Cantois.
The following (misspecified) model 'monod' is proposed to describe baker's yeast growth
'''

monod='np.array([((theta[0]*x[1]/(theta[1]+x[1]))-u[0]-theta[3])*x[0],-((theta[0]*x[1]/(theta[1]+x[1]))*x[0])/theta[2]+u[0]*(u[1]-x[1])])'

model_equations=monod # <- Assume approximated model 

'''
By uncommenting the following line, the true model equations 'cantois' are used to fit the dataset.
You will observe that, if model_equations=cantois, the goodness-of-fit test is PASSED (The model is not falsified)
Furthermore, all the Model Modification Indexes are below 1, meaning that there is no evidence to 
justify the evolution of any model parameter into a state-dependent function
'''
#model_equations=cantois #<- Assume true model


#Parameter estimation through the optimisation of the log-likelihood function

initial_guess=np.array([0.301, 0.286, 0.515, 0.043])
estimation=minimize(loglikelihood, initial_guess, args=(dataset, sigma, model_equations, ), method='Nelder-Mead') 

estimates=estimation.x    
sum_of_squared_residuals=2*loglikelihood(estimates, dataset, sigma, model_equations)

#Evaluate statistical quality of estimates

covariance_estimates=np.linalg.inv(observed_fisher(dataset,sigma, estimates, model_equations))
chi2_reference_values=st.chi2.ppf(0.05, len(dataset)*len(measurable)-len(estimates)), st.chi2.ppf(0.95, len(dataset)*len(measurable)-len(estimates))

#Print results

print('The estimated parameters are: ', estimates)
#print('The covariance of the parameter estimates is: ', covariance_estimates)
print('t-test 95% significance', t_test(estimates, covariance_estimates, 0.95, len(dataset)*len(measurable)-len(estimates)))
print('The sum of square residuals is: {0:.2f}'.format(sum_of_squared_residuals))
print('The 5% and 95% chi2 values of reference are: {0:.2f} and {1:.2f}'.format(*chi2_reference_values))


if sum_of_squared_residuals>chi2_reference_values[1]:
    print('\nThe candidate model is falsified for under-fitting.')
    print('The sum of squared residuals: {0:.2f}; is larger than the 95% chi2 value of reference: {1:.2f};'.format(sum_of_squared_residuals, chi2_reference_values[1]))


elif sum_of_squared_residuals<chi2_reference_values[0]:
    
    print('\nThe candidate model is falsified for over-fitting.')
    print('The sum of squared residuals: {0:.2f}; is smaller than the 5% chi2 value of reference: {1:.2f};'.format(sum_of_squared_residuals, chi2_reference_values[0]))

else:
    
    print('\nThe candidate model is not falsified by the Goodness-of-fit test.')
    print('The sum of squared residuals: {0:.2f}; is within the range defined by the 5% and 95% chi2 values of reference'.format(sum_of_squared_residuals))

'''
The following code computes the Model Modification Indexes associated to the model parameters.
The MMI represents a measure of model misspecification. If the MMI is above 1 for some model parameter,
then a significant improvement in the model fitting quality is expected, should that parameter be evolved
into a function of some state variables.
'''

score_statistics, score_reference = lagrange_multiplier_diagnosis(model_equations, estimates, dataset, sigma, method='univariate', gradient='default', parameter_to_diagnose='all')
model_modification_indexes=score_statistics/score_reference
print('\nThe model modification indexes are: ', model_modification_indexes)


if np.array(model_modification_indexes>1).any():
    
    MMI_max_index=np.argmax(model_modification_indexes)
    
    print('\nThe diagnosis suggests that a significant improvement of the fitting may be achieved by evolving parameter theta[{}].\n'.format(MMI_max_index))

    parameter_to_diagnose=np.array([MMI_max_index])
    effects=np.array(['x[0]', 'x[1]', '1/x[0]', '1/x[1]', 'u[0]', 'u[1]', '1/u[0]', '1/u[1]'])
    
    
    print('The Effect Relevance Indexes (ERIs) are now computed to quantify the relevance of the following candidate effects:\n'
           +', '.join(list(effects))
           +'; for the evolution of parameter theta[{}]\n'.format(MMI_max_index))
    
    first_order_eff = first_order_effects_on_parameters(model_equations, estimates, dataset, sigma, parameter_to_diagnose, effects, approach='multivariate', output='print')  
    
    print('\nThe most relevant effect for the evolution of theta[{}] '.format(MMI_max_index)
          +'is '+max(first_order_eff['theta[{}]'.format(MMI_max_index)], key=first_order_eff['theta[{}]'.format(MMI_max_index)].get)
          +'\nThe ERI associated with the most relevant effect is {0:.2f}.'.format(max(first_order_eff['theta[{}]'.format(MMI_max_index)].values())))
          
else:
    
    print('\nAll the MMIs are below 1. Hence, there is no evidence to justify the evolution of any model parameter.')