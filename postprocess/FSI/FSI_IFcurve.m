clear all; %close all; 
set(0,'defaulttextinterpreter','latex'); rng('shuffle');
%% Load data parameters
dt=0.0001; % time step in s
T=1; % Length of simulation time saving data (excluding spikes) in s
postprocess_Input = true;
postprocess_Spikes = true;
postprocess_IFcurve = true;
directory='../../graphs/Traub/' % just for feedback
fileExt='.dat';
%% Load data
if (postprocess_Input)
    fid = fopen([directory,'Wave',fileExt],'r');
    XdimInput = fread(fid, 1, 'int');
    YdimInput = fread(fid, 1, 'int');
    ZdimInput = fread(fid, 1, 'int');
    temp = fread(fid, XdimInput*YdimInput*ZdimInput, 'float');
    fclose(fid);
    clear fid;
    % 4D: time, X, Y, Z
    temp = reshape(temp, [XdimInput, YdimInput, ZdimInput, numel(temp)/(XdimInput*YdimInput*ZdimInput)]);
    input = permute(temp, [4, 1, 2, 3]);
    clear temp;
end
if (postprocess_Spikes)
    fid = fopen([directory,'Spike',fileExt],'r');
    XdimSpikes = fread(fid, 1, 'int');
    YdimSpikes = fread(fid, 1, 'int');
    ZdimSpikes = fread(fid, 1, 'int'); % have to load all sadly
    temp = fread(fid, Inf, 'int');
    fclose(fid);
    clear fid;
    % 1 row per spike, where spike(:,1) is neuron id and spike(:,2) is
    % spike time
    temp = reshape(temp, 2, numel(temp)/2)';
    temp(temp(:,2)>T/dt,:) = []; % filter out bigger than T
    temp(:,1) = temp(:,1)+1; % because of 0 indexing
    temp = sortrows(temp); % Soft by neuron id
    [~, temp_i] = unique(temp(:,1)); % First spike of each neuron (for those that have spikes)
    spikesN = circshift(temp_i,-1) - temp_i; % Number of spikes for each neuron (for those that have spikes)
    % 4D: spike times, X, Y, Z
    maxSpikes = max(spikesN);
    spike = ones(maxSpikes,XdimSpikes,YdimSpikes,ZdimSpikes)*-1;
    i=1; % Neuron id counter
    j=1; % Position in temp3
    k=1; % Position in temp4_i and spikesN
    for z=1:ZdimSpikes
        for y=1:YdimSpikes
            for x=1:XdimSpikes
                % If this neuron has spikes
                if (temp(j,1) == i)
                   spike(1:spikesN(k),x,y,z) = ...
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
%% Plot
if (postprocess_Input)
    figure(1); clf; % Input
    histogram(input);
    title('input');
    xlabel('input strength');
    ylabel('count');
    print([directory,'input'],'-dpng');
end
if (postprocess_Spikes)
    figure(2); clf; % Spikes
    Nspikes=XdimSpikes; % Number of neurons in one segment
    maxSpikes=round(maxSpikes/1); % can be used to shrink if too large
    temp = zeros(maxSpikes*Nspikes,2);
    for x=1:XdimSpikes
        temp((x-1)*maxSpikes+1:x*maxSpikes,1) ...
            = ones(maxSpikes,1)*x;
        temp((x-1)*maxSpikes+1:x*maxSpikes,2) ...
            = spike(1:maxSpikes,x,y,z);
    end
    temp(temp(:,2)==-1,:) = [];
    scatter(temp(:,2),temp(:,1));
    xlim([0*(1/dt) T*(1/dt)]);
    ylim([0 Nspikes]);
    title('Spiking activity');
    xlabel('t [dt]');
    ylabel('neuron id');
    print([directory,'spikes'],'-dpng');
    clear temp temp2 i segmentDim Nspikes;
    
    figure(3); clf; % Firing rates
    Hz = zeros(1,XdimSpikes);
    for x=1:XdimSpikes
        Hz(x) = sum(spike(1:end,x)>=0); % ignore first spike as it is an artifact
    end
    histogram(Hz);
    title('Firing rates');
    xlabel('Hz');
    ylabel('count');
    print([directory,'Hz'],'-dpng');
end
if (postprocess_IFcurve)
    figure(4); clf; % IF curve
    scatter(input, Hz);
    title('IF');
    xlabel('input');
    ylabel('Hz');
    ylim([0 400]);
    print([directory,'IFcurve'],'-dpng');
end