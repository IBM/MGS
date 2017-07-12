clear all; %close all;
set(0,'defaulttextinterpreter','latex'); rng('shuffle');
numFigs=2;
for i=1:numFigs
    figure(i); clf;
end
%% Load data parameters
sf=0.001; % sampling interval in s
loadData = true;
postprocess_Spikes = true;
postprocess_Voltages = true;
directory='../../graphs/IAF/'
fileExt='.dat';
%% Load data
if (loadData)
    if (postprocess_Spikes)
        fid = fopen([directory,'Spike',fileExt],'r');
        Xdim = fread(fid, 1, 'int');
        Ydim = fread(fid, 1, 'int');
        Zdim = fread(fid, 1, 'int');
        temp = fread(fid, Inf, 'int');
        fclose(fid);
        clear fid;
        % 1 row per spike, where spike(:,1) is neuron id and spike(:,2) is
        % spike time
        temp = reshape(temp(1:floor(numel(temp)/2)*2), ...
            2, floor(numel(temp)/2))';
        temp(:,1) = temp(:,1)+1; % because of 0 indexing
        temp = sortrows(temp); % Soft by neuron id
        [~, temp_i] = unique(temp(:,1)); % First spike of each neuron (for those that have spikes)
        spikesN = circshift(temp_i,-1) - temp_i; % Number of spikes for each neuron (for those that have spikes)
        % 4D: spike times, X, Y, Z
        maxSpikes = max(spikesN);
        spikes = ones(maxSpikes,Xdim,Ydim,Zdim)*-1;
        i=1; % Neuron id counter
        j=1; % Position in temp3
        k=1; % Position in temp4_i and spikesN
        for z=1:Zdim
            for y=1:Ydim
                for x=1:Xdim
                    % If this neuron has spikes
                    if (temp(j,1) == i)
                        spikes(1:spikesN(k),x,y,z) = ...
                            temp(temp_i(k):temp_i(k)+spikesN(k)-1,2);
                        j=j+spikesN(k);
                        k=k+1;
                    end
                    i=i+1;
                end
            end
        end
        clear temp temp_i spikesN;
    end
    if (postprocess_Voltages)
        fid = fopen([directory,'Voltage',fileExt],'r');
        Xdim = fread(fid, 1, 'int');
        Ydim = fread(fid, 1, 'int');
        Zdim = fread(fid, 1, 'int');
        temp = fread(fid, Inf, 'float');
        fclose(fid);
        clear fid;
        % 4D: time, X, Y, Z, Voltage
        temp = reshape(temp(1:floor(numel(temp)/((Xdim*Ydim*Zdim)))*Xdim*Ydim*Zdim), ...
            [Xdim, Ydim, Zdim, floor(numel(temp)/(Xdim*Ydim*Zdim))]);
        voltages = permute(temp, [4, 1, 2, 3]);
        clear temp;
    end
end
%% Plot parameters
Nspikes=Xdim; % Number of neurons to plot (a random sample without the specified range set below)
if ((exist('Xdim','var')) && (exist('Ydim','var')) && (exist('Zdim','var')))
    Xmin=1; Xmax=Xdim;
    Ymin=1; Ymax=Ydim;
    Zmin=1; Zmax=Zdim;
else
    warning('Need to load data to plot.');
    return;
end
%% Plot
Drange = [Xmax-Xmin,Ymax-Ymin,Zmax-Zmin];
points = rand(Nspikes,3);
points = bsxfun(@times,Drange,points);
Dmin = [Xmin,Ymin,Zmin];
points = round(bsxfun(@plus,Dmin,points));
if (postprocess_Spikes)
    figure(1); % Spikes
    maxSpikes=round(maxSpikes/1); % can be used to shrink if too large
    temp = zeros(maxSpikes*Nspikes,2);
    for i=1:Nspikes
        temp((i-1)*maxSpikes+1:i*maxSpikes,1) = ones(maxSpikes,1)*i;
        temp((i-1)*maxSpikes+1:i*maxSpikes,2) = spikes(1:maxSpikes,points(i,1),points(i,2),points(i,3));
    end
    scatter(temp(:,2),temp(:,1));
    clear Drange Dmin temp temp2;
end
neuron=points(1,1);
if (postprocess_Voltages)
    figure(2);
    plot(voltages(:,neuron,1,1));
end