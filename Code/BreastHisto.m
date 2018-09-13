clear; 
clc;
close all;
addpath('./Utils');
addpath('./Liblinear');
make; 

TrnSize = 200; 
ImgSize = 50; 
ImgFormat = 'gray';

load('./BreastData/1000+1000/dataset'); 
fileID = fopen('./Output/output.txt','w');
% Randnidx = randperm(size(train,1)); 
% train = train(Randnidx,:); 

TrnData = train(1:TrnSize,1:end-1)';
TrnLabels = train(1:TrnSize,end);
clear train;

TestData = test(1:20,1:end-1)';
TestLabels = test(1:20,end);
clear test;


nTestImg = length(TestLabels);


PCANet.NumStages = 2;
PCANet.PatchSize = [7 7];
PCANet.NumFilters = [8 8];
PCANet.HistBlockSize = [7 7]; 
PCANet.BlkOverLapRatio = 0.5;
PCANet;


fprintf('\n PCANet Training Start \n');
TrnData_ImgCell = mat2imgcell(TrnData,ImgSize,ImgSize,ImgFormat);
clear TrnData; 
tic;
[ftrain,V,BlkIdx] = PCANet_train(TrnData_ImgCell,PCANet,1);
PCANet_TrnTime = toc;
clear TrnData_ImgCell; 
fprintf('\n PCANet Training End \n');

fprintf('\n Training SVM Classifier Start \n');
tic;
models = train(TrnLabels, ftrain', '-s 1 -q');
LinearSVM_TrnTime = toc;
clear ftrain; 
fprintf('\n Training SVM Classifier End \n');


TestData_ImgCell = mat2imgcell(TestData,ImgSize,ImgSize,ImgFormat);
clear TestData; 

fprintf('\n Testing Start \n');

nCorrRecog = 0;
RecHistory = zeros(nTestImg,1);

tic; 
truepos = 0;
trueneg = 0;
falsepos = 0; 
falseneg = 0;
plotX=zeros(1,nTestImg);
plotY=zeros(1,nTestImg);
PredictOutput=zeros(1,nTestImg);
for idx = 1:1:nTestImg
    
    ftest = PCANet_FeaExt(TestData_ImgCell(idx),V,PCANet);

    [xLabel_est, accuracy, decision_values] = predict(TestLabels(idx),sparse(ftest'), models, '-q');
    PredictOutput(idx)=xLabel_est;
    if xLabel_est == TestLabels(idx)
        RecHistory(idx) = 1;
        nCorrRecog = nCorrRecog + 1;
    end
    
    if 1 == TestLabels(idx)
        if 1 == xLabel_est
            truepos = truepos + 1;
        end
        if 0 == xLabel_est
            falseneg = falseneg + 1;
        end
    end
    
    if 0 == TestLabels(idx)
        if 0 == xLabel_est
            trueneg = trueneg + 1;
        end
        if 1 == xLabel_est
            falsepos = falsepos + 1;
        end
    end
    
    fprintf(fileID,'Accuracy up to %d tests is %.2f%%; taking %.2f secs per testing sample on average. \r\n',[idx 100*nCorrRecog/idx toc/idx]); 
    plotX(idx)=idx;
    plotY(idx)=100*nCorrRecog/idx;
    TestData_ImgCell{idx} = [];
    
end
fprintf('\n Testing End \n');
Averaged_TimeperTest = toc/nTestImg;
Accuracy = nCorrRecog/nTestImg; 
ErRate = 1 - Accuracy;


fprintf(fileID,'\r\n Summary of PCANet and SVM classifier \r\n');
fprintf(fileID,'\r\n PCANet training time: %.2f secs.', PCANet_TrnTime);
fprintf(fileID,'\r\n SVM training time: %.2f secs.', LinearSVM_TrnTime);
fprintf(fileID,'\r\n Testing Accuracy: %.2f%%', 100*Accuracy);
fprintf(fileID,'\r\n Testing error rate: %.2f%%', 100*ErRate);
fprintf(fileID,'\r\n Average testing time %.2f secs per test sample. \r\n',Averaged_TimeperTest);
fprintf(fileID,'\r\n True Positive %d. \r\n',truepos);
fprintf(fileID,'\r\n False Positive %d. \r\n',falsepos);
fprintf(fileID,'\r\n True Negative %d. \r\n',trueneg);
fprintf(fileID,'\r\n False Negative %d. \r\n',falseneg);
fclose(fileID);
plot(plotX,plotY);
    