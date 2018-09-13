function [f, BlkIdx] = PCANet_FeaExt(InImg,V,PCANet)

addpath('./Utils')

if length(PCANet.NumFilters)~= PCANet.NumStages
    disp('Length(PCANet.NumFilters)~=PCANet.NumStages')
    return
end

NumImg = length(InImg);

OutImg = InImg; 
ImgIdx = (1:NumImg)';
clear InImg;
for stage = 1:PCANet.NumStages
     [OutImg, ImgIdx] = PCA_output(OutImg, ImgIdx, PCANet.PatchSize(stage), PCANet.NumFilters(stage), V{stage});  
end

[f, BlkIdx] = HashingHist(PCANet,ImgIdx,OutImg);





