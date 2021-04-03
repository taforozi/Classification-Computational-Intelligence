%% Fuzzy Systems - Classification (Part 1)
% Aforozi Thomais - AEM 9291
% TSK Models 
% 2 - Sub. Clustering (Class independent)
% 2 - Sub. Clustering per class (Class dependent)

%% Clear Workspace
clear all;
close all;

%% Load & Prepare & Split the data
% load haberman.data
haberman = importdata('haberman.data');
[training_data,validation_data,check_data] = split_scale(haberman,1);

%% TSK Model 1
% Sub. Clustering (Class independent)
% large radius - less rules
[fis1,OA_1,PA_1,UA_1,k_1,ErrorMatrix_1] = TSK_Model_1(training_data,validation_data,check_data,haberman);

%% TSK Model 2
% Sub. Clustering (Class independent)
% small radius - more rules
[fis2,OA_2,PA_2,UA_2,k_2,ErrorMatrix_2] = TSK_Model_2(training_data,validation_data,check_data,haberman);

%% TSK Model 3
% Sub. Clustering per class (Class dependent)
% large radius - less rules
[fis3,OA_3,PA_3,UA_3,k_3,ErrorMatrix_3] = TSK_Model_3(training_data,validation_data,check_data,haberman);

%% TSK Model 4
% Sub. Clustering per class (Class dependent)
% small radius - more rules
[fis4,OA_4,PA_4,UA_4,k_4,ErrorMatrix_4] = TSK_Model_4(training_data,validation_data,check_data,haberman);
