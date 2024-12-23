# main.py
# Example on how to run an adaptation of the pareto_selection adaptation

import stepwise_selection as ss #to select models
import os
#-----------------------------------------------------------------------------------------------------------------------
# STEP 1: Initialize selector. The most relevant flags correspond to "database_step" and "features_pareto_step".
#         Both of these are lists of lists, the first one contains the names of the databases to analyze, if it contains
#         more than one database in an element; e.g., ['ADNI', 'NACC'], it will extract the worst case for each pareto
#         feature between the various databases; e.g, if we are looking at the MAE, it would extract the largest MAE between
#         'ADNI' and 'NACC'. Simultaneously, 'features_pareto_step', allows us to specify which metrics do we want to use
#         on each database in 'database_step' as pareto_features. Observe that the 'origin' parameter allows us to mix the
#         results of each of us. SO far MC, SV available.
'''
#27-09 (MC)
settings_examples = [{'database_step':[['UKBB'],['ADNI','NACC']], 'features_pareto_step': 2*[['MAE_orx_bounded','max_MAE_bin_orx_bounded','auroc_bounded_CNvsNoCN']],
                      'directory_results' : 'pareto_fronts/test_1/'},
                     {'database_step':[['DALL'],['DALL']], 'features_pareto_step': [['Rsquared_orx_bounded','abs(corr[PAD,CA]_orx_bounded)','auroc_bounded_CNvsNoCN'],
                                                                                  ['MAE_orx_bounded','max_MAE_bin_orx_bounded','auroc_bounded_CNvsCondition.Dementia']],'directory_results' : 'pareto_fronts/test_2/'}]
'''
# Example Explanation: In the first example, for the first Pareto we use only UKBB. The remaining models are then explored
#                      in the worst case scenario between 'ADNI' and 'NACC'. In this case we use the same Pareto features
#                      in each case.
#                      In the second example, we only use 'DALL'. In this case, at each step of Pareto we use different sets of metrics
'''
#30-09 (SV)
pareto_features = [['MAE_orx_bounded','max_MAE_bin_orx_bounded','Rsquared_orx_bounded','abs(corr[PAD,CA]_orx_bounded)'],
                   ['MAE_orx_bounded','max_MAE_bin_orx_bounded','abs(corr[PAD,CA]_orx_bounded)','auroc_bounded_CNvsNoCN'],
                   ['MAE_orx_bounded','max_MAE_bin_orx_bounded','abs(corr[PAD,CA]_orx_bounded)','auroc_bounded_CNvsCondition.Parkinson'],
                   ['MAE_orx_bounded','max_MAE_bin_orx_bounded','abs(corr[PAD,CA]_orx_bounded)','auroc_bounded_CNvsCondition.MultipleSclerosis'],
                   ['MAE_orx_bounded','max_MAE_bin_orx_bounded','abs(corr[PAD,CA]_orx_bounded)','auroc_bounded_CNvsCondition.Dementia'],
                   ['MAE_orx_bounded','max_MAE_bin_orx_bounded','abs(corr[PAD,CA]_orx_bounded)','auroc_bounded_CNvsCondition.MCI'],
                   ['MAE_orx_bounded','max_MAE_bin_orx_bounded','abs(corr[PAD,CA]_orx_bounded)','auroc_bounded_CNvsCondition.SMC'],
                   ['MAE_orx_bounded','max_MAE_bin_orx_bounded','abs(corr[PAD,CA]_orx_bounded)','MAE_orx_age.[80,85]']
                   ]

education = ['MAE_orx_bounded_education.[0,7]','MAE_orx_bounded_education.[16,30]','MAE_orx_bounded_education.[8,15]']

ethnicity = ['MAE_orx_bounded_ethnicity.Mixed','MAE_orx_bounded_ethnicity.Caribbean','MAE_orx_bounded_ethnicity.Asian','MAE_orx_bounded_ethnicity.NativeAmerican','MAE_orx_bounded_ethnicity.White','MAE_orx_bounded_ethnicity.Black']

manufacturer = ['MAE_orx_bounded_manufacturer.GE','MAE_orx_bounded_manufacturer.Siemens','MAE_orx_bounded_manufacturer.Philips']

settings_examples = [{'database_step':[['UKBB'],['ADNI','NACC']], 'features_pareto_step': 2*[pareto_features[i]], 'directory_results' : 'test_cn'+str(i)} for i in [0,1,7]] + [{'database_step':[['DALL'],['ADNI','NACC']], 'features_pareto_step': 2*[pareto_features[i]],                                            'directory_results' : 'test_cog'+str(i)} for i in [4,5,6]] + [{'database_step':[['DALL'],['DALL'],['DALL'],['DALL']], 'features_pareto_step': [pareto_features[0],education,ethnicity,manufacturer],'directory_results' : 'test_pop'}] + [{'database_step':[['UKBB'],['UKBB']], 'features_pareto_step': [pareto_features[0],pareto_features[2]],'directory_results' : 'test_pd'}]+ [{'database_step':[['UKBB'],['UKBB']], 'features_pareto_step': [pareto_features[0],pareto_features[3]],'directory_results' : 'test_sm'}]

for gender in ['M','F']:
    for age_bias_exclusion in [True, False]:
        for settings in settings_examples:
            print('---------------------------------------------------------------------------------------------------')
            print('*case:',gender,str(age_bias_exclusion),str(settings))
            dp = ss.model_selection(database_step = settings['database_step'], features_pareto_step = settings['features_pareto_step'], origin = ['MC','SV'],
                                    gender = gender, age_bias_exclusion = age_bias_exclusion, directory_results = os.path.join('pareto_fronts',settings['directory_results'],gender+'_'+'abe_'+str(age_bias_exclusion)))
            dp.stepwise_pareto()
'''
origin_cases = [['SV','MC','GA'],['SV'],['MC'],['GA']]
for origin in origin_cases:
    #02-10 (MC)
    settings_examples = [{'database_step':[['UKBB'],['ADNI','NACC']], 'features_pareto_step': 2*[['MAE_orx_bounded','max_MAE_bin_orx_bounded','auroc_bounded_CNvsNoCN','generalization_max']],
                          'directory_results' : 'test_gen_1_max_'+str(origin)+'/'},
                         {'database_step': [['UKBB'], ['ADNI', 'NACC']], 'features_pareto_step': 2 * [
                             ['MAE_orx_bounded', 'max_MAE_bin_orx_bounded', 'auroc_bounded_CNvsNoCN',
                              'generalization_median']],
                          'directory_results': 'test_gen_1_median'+str(origin)+'/'},
                         {'database_step': [['UKBB'], ['ADNI', 'NACC']], 'features_pareto_step': 2 * [
                             ['MAE_orx_unbounded', 'max_MAE_bin_orx_unbounded', 'auroc_unbounded_CNvsNoCN',
                              'generalization_max']],
                          'directory_results': 'test_gen_2_max'+str(origin)+'/'},
                         {'database_step': [['UKBB'], ['ADNI', 'NACC']], 'features_pareto_step': 2 * [
                             ['MAE_orx_unbounded', 'max_MAE_bin_orx_unbounded', 'auroc_unbounded_CNvsNoCN',
                              'generalization_median']],
                          'directory_results': 'test_gen_2_median'+str(origin)+'/'},
                         {'database_step': [['UKBB'], ['ADNI', 'NACC']], 'features_pareto_step': 2 * [
                             ['MAE_orx_bounded', 'max_MAE_bin_orx_unbounded', 'auroc_unbounded_CNvsNoCN',
                              'generalization_max']],
                          'directory_results': 'test_gen_3_max'+str(origin)+'/'},
                         {'database_step': [['UKBB'], ['ADNI', 'NACC']], 'features_pareto_step': 2 * [
                             ['MAE_orx_bounded', 'max_MAE_bin_orx_unbounded', 'auroc_unbounded_CNvsNoCN',
                              'generalization_median']],
                          'directory_results': 'test_gen_3_median'+str(origin)+'/'},
                         {'database_step': [['UKBB'], ['ADNI', 'NACC']], 'features_pareto_step': [
                             ['MAE_orx_bounded', 'max_MAE_bin_orx_unbounded', 'auroc_unbounded_CNvsNoCN'],
                             ['MAE_orx_bounded', 'max_MAE_bin_orx_unbounded', 'auroc_unbounded_CNvsNoCN',
                              'generalization_max']
                         ],
                          'directory_results': 'test_gen_4_max' + str(origin) + '/'},
                         {'database_step': [['UKBB'], ['ADNI', 'NACC']], 'features_pareto_step': [
                             ['MAE_orx_bounded', 'max_MAE_bin_orx_unbounded', 'auroc_unbounded_CNvsNoCN'],
                             ['MAE_orx_bounded', 'max_MAE_bin_orx_unbounded', 'auroc_unbounded_CNvsNoCN',
                              'generalization_median']],
                          'directory_results': 'test_gen_4_median' + str(origin) + '/'},
                         {'database_step': [['DALL'], ['DALL']], 'features_pareto_step': [
                             ['Rsquared_orx_bounded','abs(corr[PAD,CA]_orx_unbounded)','auroc_unbounded_CNvsNoCN'],
                             ['MAE_orx_bounded','max_MAE_bin_orx_unbounded','auroc_bounded_CNvsCondition.Dementia',
                              'generalization_max']
                         ],
                          'directory_results': 'test_gen_5_max' + str(origin) + '/'},
                         {'database_step': [['DALL'], ['DALL']], 'features_pareto_step': [
                             ['Rsquared_orx_bounded','abs(corr[PAD,CA]_orx_unbounded)','auroc_unbounded_CNvsNoCN'],
                             ['MAE_orx_bounded','max_MAE_bin_orx_unbounded','auroc_unbounded_CNvsCondition.Dementia',
                              'generalization_median']],
                          'directory_results': 'test_gen_5_median' + str(origin) + '/'}
                         ]
    for gender in ['M','F']:
        for age_bias_exclusion in [False]:#[True, False]:
            for settings in settings_examples:
                print('---------------------------------------------------------------------------------------------------')
                print('*case:',gender,str(age_bias_exclusion),str(settings),gender)
                dp = ss.model_selection(database_step = settings['database_step'], features_pareto_step = settings['features_pareto_step'], origin = origin,
                                        gender = gender, age_bias_exclusion = age_bias_exclusion, generalization_cases = ['database', 'manufacturer','machine','ethnicity'],
                                        directory_results = os.path.join('pareto_fronts',settings['directory_results'],gender+'_'+'abe_'+str(age_bias_exclusion)))
                dp.stepwise_pareto()

