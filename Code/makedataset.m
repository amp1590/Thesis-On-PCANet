clear;clc;
test=csvread('BreastData/test.csv');
train=csvread('BreastData/train.csv');
save('./BreastData/dataset.mat','train','test');
clear;
disp("Done");