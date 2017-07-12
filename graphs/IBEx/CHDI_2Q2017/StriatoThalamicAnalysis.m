% Striato-Thalamic analysis
var = {['M1'], ['Msup'], ['S1'], ['Ssec'], ['TRN']};

% Connectivity matrix
Str2Th = zeros(nCx, nStr);
for m=1:nFrontal
    Str2Th(Cxoffs(m)+1:Cxoffs(m+1), :) = StrTh{m};
end
figure; imagesc(Str2Th);

%% Striatum firing rates and intensity of firing volleys
t1 = 601000; t2 = 610000;
figure('name', ['t=' num2str(t1)]);
subplot(2,1,1); bar(sum(hx_s(t1:t2,:)>0)); xlim([0 nStr]); ylim([0 4000]);
xlabel('Str units'); ylabel('nb ot tsteps active'); 
title('Str firings'); 
subplot(2,1,2); histogram(sum(hx_s(t1:t2,:)'>0), 'BinWidth', 1); xlim([0 30]); ylim([0 6000]);
xlabel('Nbr of Str units active / tstep'); ylabel('tsteps counts'); 
title('Distribution of intensity of Str firing volleys'); 

%% SORN firing rates and intensity of firing volleys
t1 = 504000; t2 = 505000;
figure('name', ['t=' num2str(t1)]);
for m=1:nAreas
    subplot(2,4,m); bar(sum(SFRE_raster(t1:t2,:))); xlim([1 Cxsz(m)]); ylim([0 1000]);
    xlabel([var{m} ' units']); ylabel('nb ot tsteps active'); 
    title([var{m} ' firings']); 
    subplot(2,4,nAreas+m); histogram(sum(SFRE_raster(t1:t2,:)'), 'BinWidth', 1); xlim([0 Cxsz(m)]); %ylim([0 6000]);
    xlabel(['Nbr of ' var{m} ' units active / tstep']); ylabel('tsteps counts'); 
    title(['Distribution of intensity of ' var{m} 'firing volleys']);
end

%% CV of SORN firing (CV=std/mean) per area
t1 = 570000; t2 = 580000;
figure;
CV_fr = zeros(nAreas+1,1);
mu=0;
for m=1:nAreas
    CV_fr(m) = std(sum(SFRE_raster(t1:t2,Cxoffs(m)+1:Cxoffs(m+1))))/mean(sum(SFRE_raster(t1:t2,Cxoffs(m)+1:Cxoffs(m+1))));
    mu = mu + CV_fr(m);
end
bar(CV_fr(1:4)); hold on;
CV_fr(m+1) = mean(CV_fr(1:m));
bar(m+1, CV_fr(m+1), 'r');

%% CV ISI of SORN units
figure; 
CV_ISI = zeros(nAreas+1,1);
for m=1:nAreas
    idx = find(SFRE_raster(t1:t2,Cxoffs(m)+1:Cxoffs(m+1))==1);
    subplot(1,nAreas+1,m); 
    histogram(diff(idx), 'Normalization', 'probability', 'BinWidth', 1);
    xlim([0 100]);
    CV_ISI(m) = std(diff(idx))/mean(diff(idx));
end
subplot(1,nAreas+1,m+1); bar(CV_ISI(1:m)); hold on;
bar(m+1, mean(CV_ISI(1:m)), 'r');


%% Str temporal convolution
t1 = 550000; t2 = 650000;
t_window = 1000;
half_tw = round(t_window/2);
str_epoch1=hx_s(t1-t_window:t2+t_window,:);

gauss_conv = normpdf(-half_tw:half_tw, 0, round(t_window/5));  % gaussian kernel to convolute signal (tau=1000)
boxcar = ones(1,t_window+1);
for i=1:(size(str_epoch1,1)-t_window)    
    %str_epoch1(i,:) = gauss_conv * str_epoch1(i:i+t_window,:);
    str_epoch1(i,:) = boxcar * str_epoch1(i:i+t_window,:);
end
str_epoch1=str_epoch1(half_tw+1:end-t_window-half_tw,:);
figure; 
imagesc(str_epoch1')


%% SORN temporal convolution
t1 = 550000; t2 = 650000;
t_window = 1000;
half_tw = round(t_window/2);
sorn_epoch1=SFRE_raster(t1-t_window:t2+t_window,:);

gauss_conv = normpdf(-half_tw:half_tw, 0, round(t_window/5));  % gaussian kernel to convolute signal (tau=1000)
boxcar = ones(1,t_window+1);
for i=1:(size(sorn_epoch1,1)-t_window)    
    %str_epoch1(i,:) = gauss_conv * str_epoch1(i:i+t_window,:);
    sorn_epoch1(i,:) = boxcar * sorn_epoch1(i:i+t_window,:);
end
sorn_epoch1=sorn_epoch1(half_tw+1:end-t_window-half_tw,:);
figure; 
imagesc(sorn_epoch1')

%% Str-Cx "BOLD" correlation
t1 = 600000; t2 = 800000;

% setup filter
sf = 0.001; bf=0.3; nth=4; 
sh=1/sf;
wf=2*bf./sh;
[b,a] = butter(nth, wf, 'low');

% Ctx filtering
sig = SFRE_raster(t1:t2,:);
y=filtfilt(b,a,sig);

% Str filtering
sigStr = hx_s(t1:t2,:);
yStr = filtfilt(b,a,sigStr);

% Plot filtered rasters
figure;
subplot(2,1,1); imagesc(y')
subplot(2,1,2); imagesc(yStr');

% Create whole dynamic
y_tot = cat(2,y,yStr);

% Correlation matrix between Ctx and Str
%corrMx = corr(y,yStr);

% Correlation matrix of both Ctx and Str
corrMx = corr(y_tot,y_tot);
absCorrMx = abs(corrMx);
thresh = std(reshape(absCorrMx, numel(absCorrMx), 1)); % threshold to apply on absolute correlation matrix
binCorrMx = abs(corrMx) > thresh; % binarized correlation matrix
G = graph(binCorrMx);
logDegPerNode = log(degree(G)+1);

% Plot those different variants of the correlation matrix
figure; 
subplot(2,2,1); imagesc(corrMx); xlabel('Ctx + Str units'); ylabel('Ctx + Str units'); title('Raw Correlation Coefficient'); % raw
subplot(2,2,2); imagesc(absCorrMx); xlabel('Ctx + Str units'); ylabel('Ctx + Str units'); title('Absolute Correlation Coefficient'); % absolute values
subplot(2,2,3); imagesc(binCorrMx); xlabel('Ctx + Str units'); ylabel('Ctx + Str units'); title('Binerized Correlation Coefficient'); % binarized
subplot(2,2,4); plot(logDegPerNode, '.', 'MarkerSize', 10); xlabel('Ctx + Str units'); ylabel('log(degree+1)'); title('Log degree value');

figure; imagesc(absCorrMx(1:400,401:500)); xlabel('Str units'); ylabel('Ctx units'); title('Absolute Correlation Coefficient'); caxis([0 1]);



