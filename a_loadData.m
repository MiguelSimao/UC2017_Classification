clear
close all

%% DATASET DIRECTORIES

dir_dataset_sg = './dataset/SG24_dataset.h5';
dir_dataset_dg = './dataset/DG10_dataset.h5';

%% LOAD STATIC GESTURE DATA
% Load from .h5 file
% Shape is (sample,variable)

X = h5read(dir_dataset_sg,'/Predictors');
T = h5read(dir_dataset_sg,'/Target');
U = h5read(dir_dataset_sg,'/User');

SG.X = X;
SG.T = T;
SG.U = U;

%% LOAD DYNAMIC GESTURE DATA
% Load from .h5 file
% Shape is (sample,variable,time)


X = h5read(dir_dataset_dg,'/Predictors');
T = h5read(dir_dataset_dg,'/Target');
U = h5read(dir_dataset_dg,'/User');

DG.X = X;
DG.T = T;
DG.U = U;

%% CLEAR UNECESSARY VARIABLES

clearvars -except SG DG
