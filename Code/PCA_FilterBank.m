function V = PCA_FilterBank(InImg, PatchSize, NumFilters) 


addpath('./Utils')


ImgZ = length(InImg);
MaxSamples = 100000;
NumRSamples = min(ImgZ, MaxSamples); 
RandIdx = randperm(ImgZ);
RandIdx = RandIdx(1:NumRSamples);

NumChls = size(InImg{1},3);
Rx = zeros(NumChls*PatchSize^2,NumChls*PatchSize^2);

for i = RandIdx 
    im = im2col_mean_removal(InImg{i},[PatchSize PatchSize]);
    Rx = Rx + im*im'; 
end
Rx = Rx/(NumRSamples*size(im,2));
[E, D] = eig(Rx);
[~, ind] = sort(diag(D),'descend');
V = E(:,ind(1:NumFilters)); 



 



