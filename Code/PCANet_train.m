function [f, V, BlkIdx] = PCANet_train(InImg,PCANet,IdtExt)

addpath('./Utils')


if length(PCANet.NumFilters)~= PCANet.NumStages
    disp('Length(PCANet.NumFilters)~=PCANet.NumStages')
    return
end

NumImg = length(InImg);

V = cell(PCANet.NumStages,1); 
OutImg = InImg; 
ImgIdx = (1:NumImg)';
clear InImg; 

for stage = 1:PCANet.NumStages
    disp(['Calcuating PCA filter bank and its outputs at stage ' num2str(stage) '........'])
    
    V{stage} = PCA_FilterBank(OutImg, PCANet.PatchSize(stage), PCANet.NumFilters(stage));
    
    if stage ~= PCANet.NumStages
        [OutImg, ImgIdx] = PCA_output(OutImg, ImgIdx, ...
            PCANet.PatchSize(stage), PCANet.NumFilters(stage), V{stage});  
    end
end

if IdtExt == 1 % enable feature extraction
    
    f = cell(NumImg,1);
    
    for idx = 1:NumImg
        if 0==mod(idx,100) 
            disp(['Extracting PCANet feature of the ' num2str(idx) 'th training sample........']); 
        end
        OutImgIndex = ImgIdx==idx; % image "idx" er feature map  
        
        [OutImg_i, ImgIdx_i] = PCA_output(OutImg(OutImgIndex), ones(sum(OutImgIndex),1), PCANet.PatchSize(end), PCANet.NumFilters(end), V{end}); 
        
        [f{idx}, BlkIdx] = HashingHist(PCANet,ImgIdx_i,OutImg_i);
        OutImg(OutImgIndex) = cell(sum(OutImgIndex),1); 
       
    end
    f = sparse([f{:}]);
    
else
    f = [];
    BlkIdx = [];
end







