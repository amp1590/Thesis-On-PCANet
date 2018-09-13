function [f, BlkIdx] = HashingHist(PCANet,ImgIdx,OutImg)

addpath('./Utils')


NumImg = max(ImgIdx);
f = cell(NumImg,1);
map_weights = 2.^((PCANet.NumFilters(end)-1):-1:0);
for Idx = 1:NumImg
    
    Idx_span = find(ImgIdx == Idx);
    NumOs = length(Idx_span)/PCANet.NumFilters(end);
    Bhist = cell(NumOs,1);
    
    for i = 1:NumOs
        
        T = 0;
        nuhash=PCANet.NumFilters(end)*(i-1) + 1;
        rezaul=Idx_span(PCANet.NumFilters(end)*(i-1) + 1);
        sakib_reza=OutImg{Idx_span(PCANet.NumFilters(end)*(i-1) + 1)};
        ImgSize = size(OutImg{Idx_span(PCANet.NumFilters(end)*(i-1) + 1)});
        for j = 1:PCANet.NumFilters(end)
            temp1=Idx_span(PCANet.NumFilters(end)*(i-1)+j);
            temp2=OutImg{Idx_span(PCANet.NumFilters(end)*(i-1)+j)};
            temp3=Heaviside(OutImg{Idx_span(PCANet.NumFilters(end)*(i-1)+j)});
            temp4=map_weights(j)*Heaviside(OutImg{Idx_span(PCANet.NumFilters(end)*(i-1)+j)});
            T = T + map_weights(j)*Heaviside(OutImg{Idx_span(PCANet.NumFilters(end)*(i-1)+j)});
            % double to decimal number conversion
            
            OutImg{Idx_span(PCANet.NumFilters(end)*(i-1)+j)} = [];
        end
        
        
        if isempty(PCANet.HistBlockSize)
            NumBlk = ceil((PCANet.ImgBlkRatio - 1)./PCANet.BlkOverLapRatio) + 1;
            HistBlockSize = ceil(size(T)./PCANet.ImgBlkRatio);
            OverLapinPixel = ceil((size(T) - HistBlockSize)./(NumBlk - 1));
            NImgSize = (NumBlk-1).*OverLapinPixel + HistBlockSize;
            Tmp = zeros(NImgSize);
            Tmp(1:size(T,1), 1:size(T,2)) = T;
            Bhist{i} = sparse(histc(im2col_general(Tmp,HistBlockSize,...
                OverLapinPixel),(0:2^PCANet.NumFilters(end)-1)'));
        else
            
            stride = round((1-PCANet.BlkOverLapRatio)*PCANet.HistBlockSize);
            histtemp1=im2col_general(T,PCANet.HistBlockSize,stride);
            histtemp2=(0:2^PCANet.NumFilters(end)-1)';
            histtemp3=histc(im2col_general(T,PCANet.HistBlockSize,stride),(0:2^PCANet.NumFilters(end)-1)');
            blkwise_fea = sparse(histc(im2col_general(T,PCANet.HistBlockSize,stride),(0:2^PCANet.NumFilters(end)-1)'));
            % calculate histogram for each local block in "T"
            
            
            blkwise_fea = bsxfun(@times, blkwise_fea,2^PCANet.NumFilters(end)./sum(blkwise_fea));
            
            
            Bhist{i} = blkwise_fea;
        end
        
    end
    f{Idx} = vec([Bhist{:}]');
    
end
f = [f{:}];

BlkIdx = kron(ones(NumOs,1),kron((1:size(Bhist{1},2))',ones(size(Bhist{1},1),1)));


