# STATISTICAL-ANALYSIS IMPLEMENTATION

Implementation used in Section 4.2 to select to perform the discussed statistical tests. In /input_files one can find all available results for different type of approaches in a standardized format and, in /individual_post_analysis, we show the obtained results distinguishing by modelling approach (MC: Machine Learning, GA: Deep Learning, SV: Modifications of Cole's Lasso).

Please refer to main.py to check details on execution and tested cases.

There are mainly two type of outcomes in each analysis:

 1) One numerical which corresponds to statistical tests. These will be saved in the corresponding .../stats/ directory.
 The .csv files correspond to individual statistical tests summaries, where the naming of the file describes the settings
 of the comparison, these are of the form:

    healthy_population + '_' + correction + '_' + bounded/unbounded metrics + metric of interest + group of interest,

 Example:
 healthy_population = 'orx'
 correction = 'none' (Please take into account that if 'None' -not 'none'- appears we analyze all correction cases simultaneously).
 bounded/unbounded = just a string to highlight whether metrics are or not bounded, not needed anymore as column names are explicit.
 metric of interest = 'MAE_orx_bounded"
 group of interest = 'Training_healthy_group"

 In addition to the .csv files, we can also encounter .txt files with the PostHoc stat tests outcomes with the same naming approach.
 the only difference is that at the beginning of the name the type of PostHoc is specified (e.g; 'anova_tukey_'...)

 Keep in mind that it is not needed to look into all .csv files separately, once all analysis are finalised, we collect
 all stat tests outcomes (except the PostHocs) in a sigle .csv that will be placed in 'post_analysis_standardized/combined_data_tests.csv' (see STEP 3)

 2) There is also one graphical outcome that will be saved in the corresponding .../plots/ directory.
 The .png files has a simpler naming convention as upto 6 "groups of interest" are gathered per plot:

    healthy_population + '_' + correction + '_' + bounded/unbounded metrics,

 Example:
 healthy_population = 'orx'
 correction = 'none' (Please take into account that if 'None' -not 'none'- appears we analyze all correction cases simultaneously).
 bounded/unbounded = just a string to highlight whether metrics are or not bounded, not needed anymore as column names are explicit.

 An example of the analysis will be saved in the drive at https://drive.google.com/file/d/1BcAHtnDxQhIHb8rVbyA4rXsZgLIZhRyZ/view?usp=drive_link

# ANALYSIS MODES. 
There are 3 options ('standard','standard_linear' and 'cole_resample').
 1. 'standard' performs the comparisons mentioned n in Section 4.1 of the manuscript.
 2. 'standard_linear' is the same but constrained to linear models (due to some numerical issues with Zhang on non-linear, we habilitated this option for an additional analysis).
 3. 'cole_resample' is tailored to comparing Cole-corrected models vs uncorrected models using resample as preprocessing step.
