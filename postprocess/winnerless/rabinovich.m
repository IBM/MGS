clear all; %close all;
% Parameters
weightFilename='../../graphs/winnerless/MSN_LN_weights_1.dat';
activityFilename='../../graphs/winnerless/MSNs.dat';
binary=true; % if binary, assumes 3D, if not binary, assumes 2D.
T=1000000;
%% Check for assymetry or not and work out connection fractrion and probability 
% of two neurons being connected.
% Load initial weight matrix
formatSpec = '%f%f%f%[^\n\r]';
fid = fopen(weightFilename,'r');
dataArray = textscan(fid, formatSpec);
fclose(fid);
MSNLNWeights1 = [dataArray{1:end-1}];
clearvars formatSpec fid dataArray ans;
S=spconvert(MSNLNWeights1);
assymetric=true;
nW=size(S,1);
for i=1:nW
    for j=1:i
        if (S(i,j)~=0 && S(j,i)~=0)
            assymetric=false;
        end
    end
end
if (assymetric)
    disp('Matrix is assymetric');
else
    disp('Matrix is not assymetric');
end
disp(['Connection fraction is ', ...
    num2str(size(MSNLNWeights1,1)/(nW^2))]);
% Probability of any two neurons being connected
n=0;
for i=1:nW
    for j=1:i
        if (S(i,j)~=0 || S(j,i)~=0)
            n=n+1;
        end
    end
end
disp(['Probability of any two neurons being connected is ', ...
    num2str(n/((nW^2)/2))]);
% Plot weight matrix
figure(1); clf;
imagesc(S);
%% Visualize the output of RabinovichWinnerlessUnit.gsl
if (binary)
    s = dir(activityFilename);
    s.bytes;
    fid = fopen(activityFilename,'r');
    Xdim = fread(fid, 1, 'int');
    Ydim = fread(fid, 1, 'int');
    Zdim = fread(fid, 1, 'int');
    numBursts = ((s.bytes - (3*4)) / 4) / 3;
    temp = zeros(numBursts,3); % [neuron id, time, activity]
    for i=1:numBursts
        temp(i,1) = fread(fid, 1, 'int')+1;
        temp(i,2) = fread(fid, 1, 'uint');
        temp(i,3) = fread(fid, 1, 'float');
    end
    activity = spconvert(temp);
    fclose(fid);
    clear s fid temp;
else
    fid = fopen(activityFilename);
    line = fgets(fid);
    dim = cell2mat(cellfun(@str2num, strsplit(line,' '), 'un', 0));
    activity=zeros(dim(1)*dim(2), T);
    line = fgets(fid);
    while ischar(line)
        line = fgets(fid);
        if (~ischar(line))
            break;
        end
        t=str2num(line);
        for i=1:dim(1)
            line = fgets(fid);
            activity(((i-1)*dim(1))+1:(i*dim(1)),t) = ...
                cell2mat(cellfun(@str2num, strsplit(line,' '), 'un', 0));
        end
        line = fgets(fid);
    end
    fclose(fid);
    clear fid;
end
% Plot activity
figure(2); clf;
imagesc(activity);