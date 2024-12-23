# main.py
# Example on how to run post_analysis implementation

import post_analysis_standardized as pas #to perform post-analysis

# OBSERVATIONS ON THE OUTCOMES:

# There are mainly two type of outcomes in each analysis:
#
# 1) One numerical which corresponds to statistical tests. These will be saved in the corresponding .../stats/ directory.
# The .csv files correspond to individual statistical tests summaries, where the naming of the file describes the settings
# of the comparison, these are of the form:
#
#    healthy_population + '_' + correction + '_' + bounded/unbounded metrics + metric of interest + group of interest,
#
# Example:
# healthy_population = 'orx'
# correction = 'none' (Please take into account that if 'None' -not 'none'- appears we analyze all correction cases simultaneously).
# bounded/unbounded = just a string to highlight whether metrics are or not bounded, not needed anymore as column names are explicit.
# metric of interest = 'MAE_orx_bounded"
# group of interest = 'Training_healthy_group"
#
# In addition to the .csv files, we can also encounter .txt files with the PostHoc stat tests outcomes with the same naming approach.
# the only difference is that at the beginning of the name the type of PostHoc is specified (e.g; 'anova_tukey_'...)
#
# Keep in mind that it is not needed to look into all .csv files separately, once all analysis are finalised, we collect
# all stat tests outcomes (except the PostHocs) in a sigle .csv that will be placed in 'post_analysis_standardized/combined_data_tests.csv' (see STEP 3)
#
# 2) There is also one graphical outcome that will be saved in the corresponding .../plots/ directory.
# The .png files has a simpler naming convention as upto 6 "groups of interest" are gathered per plot:
#
#    healthy_population + '_' + correction + '_' + bounded/unbounded metrics,
#
# Example:
# healthy_population = 'orx'
# correction = 'none' (Please take into account that if 'None' -not 'none'- appears we analyze all correction cases simultaneously).
# bounded/unbounded = just a string to highlight whether metrics are or not bounded, not needed anymore as column names are explicit.

# An example of the analysis will be saved in the drive at https://drive.google.com/file/d/1CxFwWgjVzlT68NiIRsWKJASu-KXRIsR_/view?usp=drive_link
#-----------------------------------------------------------------------------------------------------------------------
# STEP 1: Define analysis_mode. There are 4 options ('standard','standard_linear', 'cole_resample' and 'generalization).
# 1. 'standard' performs the comparisons that I mention in Section 4.1 of the paper draft.
# 2. 'standard_linear' is the same but constrained to linear models (due to some numerical issues with Zhang on non-linear,
# we habilitated this option for an additional analysis).
# 3. 'cole_resample' is tailored to comparing Cole-corrected models vs uncorrected models using resample as preprocessing step.
# 4. 'generalization' allows comparing the MAEs per groups of interest (database, ethnicity, machine manufacturer, etc) per
# model feature

for origin in ['SV','MC','GA']:
    if origin == 'MC':
        analysis_options = ['standard','standard_linear']
    else:
        analysis_options = ['standard']
    #-----------------------------------------------------------------------------------------------------------------------
    # STEP 2: Run 'standard' cases.

    for analysis_mode in analysis_options:
        for db in ['UKBB', 'ADNI', 'NACC', 'DALL']:#['DALL']: #

            if db in ['UKBB', 'DALL']: # In ADNI and NACC 'orx' == 'cole'
                healthy_groups = ['orx', 'cole']
            else:
                healthy_groups = ['orx']

            for gender in ['M','F']:
                data_location = 'input_files/metrics_g'+gender+'_d'+db+'_c'+origin+'.csv'
                directory_results = 'post_analysis_standardized_'+origin+'/'+db+'_'+gender+'_'+origin+'/'
                for age_bounded in [True,False]:
                    for type_of_approach in ['none', 'lange', 'cole', 'zhang',None]:
                        for healthy_definition in healthy_groups:
                            dp = pas.plots_individual_UKBB(data_location = data_location, directory_results = directory_results,
                                            type_of_approach = type_of_approach, healthy_definition = healthy_definition, age_bounded=age_bounded,
                                            binary_cases=['CNvsNoCN','CNvsCondition.MultipleSclerosis','CNvsCondition.Dementia','CNvsCondition.MCI','CNvsCondition.SMC','CNvsCondition.Parkinson','CNvsCondition.OtherSpinalChord',
                                                          'CNvsCondition.Alzheimer','CNvsCondition.Dementia.Alzheimer','CNvsCondition.OtherSpinalChord.Dementia','CNvsCondition.OtherSpinalChord.MultipleSclerosis',
                                                          'CNvsCondition.OtherSpinalChord.Parkinson','CNvsCondition.Alzheimer.Dementia'], analysis_mode=analysis_mode, name_origin=origin, gender = gender)
                            dp.distribution_boxplots(feature_set=['MAE_'+dp.healthy_definition+'_'+dp.age_bounded, 'max_MAE_bin_'+dp.healthy_definition+'_'+dp.age_bounded],
                                                     subfolder = '/mae/') # OBSERVATION: In distribution_boxplots it is not recommended to include "features_set" with more than two metrics at the time fpr visualization
                            #The previous dp.distribution_boxplots() call generates the outcomes described in section 4.1 (MAE), below there are other options available:
                            '''
                            dp.distribution_boxplots(feature_set=['mean_abs_meanPAD_'+dp.healthy_definition+'_'+dp.age_bounded, 'max_abs_meanPAD_'+dp.healthy_definition+'_'+dp.age_bounded],
                                                     subfolder = '/meanPAD/')'''
                            if analysis_mode in ['standard','standard_linear']:
                                dp.distribution_boxplots(['corr[BA,CA]_'+dp.healthy_definition+'_'+dp.age_bounded, '1-abs(corr[PAD,CA]_' + dp.healthy_definition + '_' + dp.age_bounded + ')'],
                                                         subfolder = '/correlation/')
                                if healthy_definition == 'orx':
                                    for binary_case in dp.available_binary_cases:
                                        dp.distribution_boxplots(feature_set=['aurocBA_' + dp.age_bounded + '_' + binary_case, 'aurocPAD_' + dp.age_bounded + '_' + binary_case],
                                                                 subfolder='/condition/'+ binary_case + '/')
    #-----------------------------------------------------------------------------------------------------------------------
    # STEP 2: Run 'cole_resample' cases. For simplicity, these cases are evaluated independently as various type_of_approach
    # are considered simultaneously in the analysis.

    for analysis_mode in ['cole_resample']:
        for db in ['UKBB', 'ADNI', 'NACC', 'DALL']:

            if db in ['UKBB', 'DALL']:
                healthy_groups = ['orx', 'cole']
            else:
                healthy_groups = ['orx']

            for gender in ['M','F']:
                data_location = 'input_files/metrics_g'+gender+'_d'+db+'_c'+origin+'.csv'
                directory_results = 'post_analysis_standardized_'+origin+'/'+db+'_'+gender+'_'+origin+'/'
                for age_bounded in [True,False]:
                    for healthy_definition in healthy_groups:
                        for group_feature in ['Description','method_type']:
                            dp = pas.plots_individual_UKBB(data_location = data_location, directory_results = directory_results,
                                            healthy_definition = healthy_definition, age_bounded=age_bounded, analysis_mode=analysis_mode, name_origin=origin)
                            dp.cole_resample(group_feature = group_feature)
    #-----------------------------------------------------------------------------------------------------------------------
    # STEP 3: Collect all statistical tests outcomes into a single file ('post_analysis_standardized/combined_data_tests.csv')
    dp.collector()
    #-----------------------------------------------------------------------------------------------------------------------
'''
# STEP 4: Run 'generalization' cases.
for analysis_mode in ['generalization']:
    for db in ['DALL']:#['ADNI', 'UKBB', 'NACC', 'DALL']:
        generalization_groups = ['ethnicity', 'CN_type', 'education_years','manufacturer', 'machine']

        if db == 'DALL':
            generalization_groups = ['database', 'manufacturer', 'machine', 'ethnicity', 'CN_type', 'education_years']

        if db in ['UKBB', 'DALL']:
            healthy_groups = ['orx', 'cole']
        else:
            healthy_groups = ['orx']

        for generalization_group in generalization_groups:
            for gender in ['M','F']:
                data_location = 'input_files/metrics_g'+gender+'_d'+db+'_c'+origin+'.csv'
                directory_results = 'generalization/'+db+'_'+gender+'_'+origin+'/'
                for age_bounded in [True,False]:
                    for type_of_approach in ['none', 'lange', 'cole', 'zhang',None]:
                        model_feature_set = ['Description', 'Training_healthy_group', 'Training_sex', 'Oversampling', 'features_type']
                        if type_of_approach is None:
                            model_feature_set.append(
                                'Age_bias_correction')  # If type_of_approach is None; i.e., I do not se select am age-bias type. I compare between age-bias approaches
                        else:
                            model_feature_set.append(
                                'method_type')  # If type_of_approach is not None, I compare linear vs non-linear approaches
                        for healthy_definition in healthy_groups:
                            for model_feature in model_feature_set:
                                dp = pas.plots_individual_UKBB(data_location=data_location,
                                                               directory_results=directory_results,
                                                               type_of_approach=type_of_approach,
                                                               healthy_definition=healthy_definition,
                                                               age_bounded=age_bounded,
                                                               analysis_mode=analysis_mode,
                                                               generalization_group=generalization_group)
                                dp.generalization(model_feature)
#-----------------------------------------------------------------------------------------------------------------------
# STEP 5: Collect all statistical tests -on generalization- outcomes into a single file ('generalization/combined_data_tests.csv')
dp.collector()'''