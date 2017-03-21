clear all; %close all;
set(0,'defaulttextinterpreter','latex'); rng('shuffle');
numFigs=23;
for i=1:numFigs
    figure(i); clf;
end
clear i;
%% Parameters
dt=0.0001;
% sf=dt; % sampling interval in s
% sf=0.0010; % sampling interval in s
sf=0.0020; % sampling interval in s
% sf=0.0100; % sampling interval in s
T=400; % Length of simulation time saving data (excluding spikes) 
       % in s to load and process
Tmin=0; Tmax=T; % Time window to load and plot
HzSf=1; % window (and step) to calculate Hz in s
cbMax=100; % scaling of color bar for imagesc plots
loadData = true;
parameterSearch = false;
postprocess_PreIndexs = true;
postprocess_InputSpikes = true;
postprocess_InputSpikesFilter = true;
inputSpikeTimesTmin=349;
inputSpikeTimesTmax=T;
postprocess_Glutamate = false;
postprocess_AvailableGlutamate = true;
postprocess_Cb1R = false;
postprocess_CleftGlutamate = true;
postprocess_AMPA = false;
postprocess_AMPAWeights = true;
postprocess_mGluR = false;
postprocess_Ca = false;
postprocess_ECB = false;
postprocess_OutputSpikes = false;
postprocess_OutputSpikesFilter = false;
outputSpikeTimesTmin=0;
outputSpikeTimesTmax=T;
postprocess_Means = true;
postprocess_Headroom = true;
HzMax=160; % limit for input Hz axes
AMPAMax=1.5; % limit for AMPA axes
finalHzT=349:399; % time window to calculate mean final Hz in s
finalHeadroomT=349:399; % time window to calculate mean final headroom in s
finalCleftGlutamateT=349:399; % time window to sum cleft clutamate in s
postprocess_RiskySynapses = true;
headroomRisk=0.4; % the headroom demarcation line for risky synapses
timeToHeadroomRisk=60000; % the time to headroom demarcation line for risky synapses in sf
postprocess_Perturbation = false;
postprocess_PerturbationHz = true;
perturbationTstart = 300; % time of the perturbation in s
perturbationTend = 600; % time of the perturbation in s
finalHeadroomTperturbation=550:600; % time window to calculate mean final headroom in s
directory='../../graphs/CorticoStriatal/';
fileExt='.dat';
% for no parameter search just set to all ranges to zero
vX_range=0;
vX_precision='.0'; % precision after decimal place
vY_range=0;
vY_precision='.0';
t_range=0;
if (parameterSearch && postprocess_Headroom)
    headroomMean = zeros(numel(vX_range),numel(vY_range),numel(t_range));
    headroomRange = zeros(numel(vX_range),numel(vY_range),numel(t_range));
    headroomSum = zeros(numel(vX_range),numel(vY_range),numel(t_range));
end
%%
for vX_r=vX_range
    for vY_r=vY_range
        for t_r=t_range
            %% Setup the directory and check if it is present
            if (parameterSearch)
                directory=[strrep(num2str(vX_r,['%',vX_precision,'f\n']),'.','_'),'/',...
                    strrep(num2str(vY_r,['%',vY_precision,'f\n']),'.','_'),'/',...
                    num2str(t_r),'/']
                for i=1:numFigs
                    figure(i); clf;
                end
                clear i;
            else
                directory % just for feedback
            end
            if (postprocess_InputSpikes && ~postprocess_PreIndexs)
                warning('To plot spikes, postprocess_PreIndexs must be turned on.');
                return
            end
            if (postprocess_Means && ...
                    (~postprocess_AvailableGlutamate || ~postprocess_AMPAWeights))
                warning(['To plot means, both available glutamate and ',...
                    ' AMPA weights must be postprocessed.']);
                return
            end
            if (postprocess_Means && ~postprocess_Headroom)
                warning('To plot headroom, postprocess_Means must be turned on.');
                return
            end
            if (postprocess_RiskySynapses && ~postprocess_Headroom)
                warning('To process risky synapses, postprocess_Headroom must be turned on.');
                return
            end
            if (postprocess_Perturbation && ~postprocess_RiskySynapses)
                warning('To process perturbation, postprocess_RiskySynapses must be turned on.');
                return
            end
            if (postprocess_Perturbation && ( ...
                    (perturbationT <= finalHzT(end)) ...
                    || (perturbationT <= finalHeadroomT(end)) ...
                    || (perturbationT <= finalCleftGlutamateT(end)) ...
                    ))
                warning('Window to calculate final headroom factors should be less than the perturbation time.');
                return
            end
            if (postprocess_Perturbation && ...
                    (perturbationT >= finalHeadroomTperturbation(1)))
                warning('Window to calculate final headroom factors under perturbation should be greater than perturbation time.');
                return
            end
                    
            %% Load data
            if (loadData)
                %%
                if (postprocess_InputSpikes)
                    fid = fopen([directory,'PoissonSpikes',fileExt],'r');
                    Xdim = fread(fid, 1, 'int');
                    Ydim = fread(fid, 1, 'int');
                    Zdim = fread(fid, 1, 'int');
                    % temp(:,1) is neuron id and temp(:,2) is spike time
                    temp = fread(fid, Inf, 'int'); % have to load all sadly
                    temp = reshape(temp, [2, numel(temp)/2]);
                    fclose(fid);
                    clear fid;
                    if (postprocess_InputSpikesFilter)
                        temp(:,temp(2,:)>inputSpikeTimesTmax/dt) = []; % filter out greater than Tmin
                        temp(:,temp(2,:)<inputSpikeTimesTmin/dt) = []; % filter out less than Tmax
                    else
                        temp(:,temp(2,:)>T/dt) = []; % filter out greater than T
                    end
                    temp(1,:) = temp(1,:)+1; % because of 0 indexing
                    temp = sortrows(temp'); % Soft by neuron id
                    [~, temp_i] = unique(temp(:,1)); % First spike of each neuron (for those that have spikes)
                    spikesN = circshift(temp_i,-1) - temp_i; % Number of spikes for each neuron (for those that have spikes)
                    spikesN(end) = size(temp,1) - sum(spikesN(1:end-1)); % fix last neuron
                    % {3D cell: X, Y, Z} (1D: spike times)
                    inputSpike = cell(Xdim,Ydim,Zdim);
                    i=1; % Neuron id counter
                    k=1; % Position in temp_i and spikesN
                    for z=1:Zdim
                        for y=1:Ydim
                            for x=1:Xdim
                                % If this neuron has spikes
                                if (temp(temp_i(k),1) == i)
                                    inputSpike{x,y,z} = ...
                                        temp(temp_i(k):temp_i(k)+spikesN(k)-1,2);
                                    k=k+1;
                                end
                                i=i+1;
                            end
                        end
                    end
                    clear temp temp_i spikesN i k z y x;
                end
                %%
                if (postprocess_Glutamate)
                    fid = fopen([directory,'Glutamate',fileExt],'r');
                    XdimInner = fread(fid, 1, 'int');
                    YdimInner = fread(fid, 1, 'int');
                    ZdimInner = fread(fid, 1, 'int');
                    glutamate = fread(fid, (XdimInner*YdimInner*ZdimInner)*(T/sf), 'float');
                    fclose(fid);
                    clear fid;
                    % 4D: time, X, Y, Z
                    glutamate = reshape(glutamate(1:floor(numel(glutamate)/((XdimInner*YdimInner*ZdimInner)))*XdimInner*YdimInner*ZdimInner), ...
                        [XdimInner, YdimInner, ZdimInner, floor(numel(glutamate)/(XdimInner*YdimInner*ZdimInner))]);
                    glutamate = permute(glutamate, [4, 1, 2, 3]);
                end 
                %%
                if (postprocess_AvailableGlutamate)
                    fid = fopen([directory,'AvailableGlutamate',fileExt],'r');
                    XdimInner = fread(fid, 1, 'int');
                    YdimInner = fread(fid, 1, 'int');
                    ZdimInner = fread(fid, 1, 'int');
                    availableGlutamate = fread(fid, (XdimInner*YdimInner*ZdimInner)*(T/sf), 'float');
                    fclose(fid);
                    clear fid;
                    % 4D: time, X, Y, Z
                    availableGlutamate = reshape(availableGlutamate(1:floor(numel(availableGlutamate)/((XdimInner*YdimInner*ZdimInner)))*XdimInner*YdimInner*ZdimInner), ...
                        [XdimInner, YdimInner, ZdimInner, floor(numel(availableGlutamate)/(XdimInner*YdimInner*ZdimInner))]);
                    availableGlutamate = permute(availableGlutamate, [4, 1, 2, 3]);
                end 
                %%
                if (postprocess_PreIndexs)
                    fid = fopen([directory,'Indexs',fileExt],'r');
                    % 4D: [1=pre 2=post], X, Y, Z
                    preIndexs = fread(fid, [2, Inf], 'int');
                    preIndexs1D = preIndexs(1,:,:,:);
                    fclose(fid);
                    clear fid;
                    preIndexs = reshape(preIndexs, [2, XdimInner, YdimInner, ZdimInner]);
                end
                %%
                if (postprocess_Cb1R)
                    fid = fopen([directory,'Cb1R',fileExt],'r');
                    XdimInner = fread(fid, 1, 'int');
                    YdimInner = fread(fid, 1, 'int');
                    ZdimInner = fread(fid, 1, 'int');
                    Cb1R = fread(fid, (XdimInner*YdimInner*ZdimInner)*(T/sf), 'float');
                    fclose(fid);
                    clear fid;
                    % 4D: time, X, Y, Z
                    Cb1R = reshape(Cb1R(1:floor(numel(Cb1R)/((XdimInner*YdimInner*ZdimInner)))*XdimInner*YdimInner*ZdimInner), ...
                        [XdimInner, YdimInner, ZdimInner, floor(numel(Cb1R)/(XdimInner*YdimInner*ZdimInner))]);
                    Cb1R = permute(Cb1R, [4, 1, 2, 3]);
                end 
                %%
                if (postprocess_CleftGlutamate)
                    fid = fopen([directory,'CleftGlutamate',fileExt],'r');
                    XdimInner = fread(fid, 1, 'int');
                    YdimInner = fread(fid, 1, 'int');
                    ZdimInner = fread(fid, 1, 'int');
                    cleftGlutamate = fread(fid, (XdimInner*YdimInner*ZdimInner)*(T/sf), 'float');
                    fclose(fid);
                    clear fid;
                    % 4D: time, X, Y, Z
                    cleftGlutamate = reshape(cleftGlutamate(1:floor(numel(cleftGlutamate)/((XdimInner*YdimInner*ZdimInner)))*XdimInner*YdimInner*ZdimInner), ...
                        [XdimInner, YdimInner, ZdimInner, floor(numel(cleftGlutamate)/(XdimInner*YdimInner*ZdimInner))]);
                    cleftGlutamate = permute(cleftGlutamate, [4, 1, 2, 3]);
                end 
                %%
                if (postprocess_AMPA)
                    fid = fopen([directory,'AMPA',fileExt],'r');
                    XdimInner = fread(fid, 1, 'int');
                    YdimInner = fread(fid, 1, 'int');
                    ZdimInner = fread(fid, 1, 'int');
                    AMPA = fread(fid, (XdimInner*YdimInner*ZdimInner)*(T/sf), 'float');
                    fclose(fid);
                    clear fid;
                    % 4D: time, X, Y, Z, 
                    AMPA = reshape(AMPA(1:floor(numel(AMPA)/((XdimInner*YdimInner*ZdimInner)))*XdimInner*YdimInner*ZdimInner), ...
                        [XdimInner, YdimInner, ZdimInner, floor(numel(AMPA)/(XdimInner*YdimInner*ZdimInner))]);
                    AMPA = permute(AMPA, [4, 1, 2, 3]);
                end
                %%
                if (postprocess_AMPAWeights)
                    fid = fopen([directory,'AMPAWeights',fileExt],'r');
                    % 3D: X, Y, Z
                    AMPAWeights1D = fread(fid, Inf, 'float');
                    fclose(fid);
                    clear fid;
                    AMPAWeights = reshape(AMPAWeights1D, [XdimInner, YdimInner, ZdimInner]);
                end
                %%
                if (postprocess_mGluR)
                    % mGluR modulation function only first
                    fid = fopen([directory,'mGluRmodulation',fileExt],'r');
                    mGluRmodulation = fread(fid, Inf, 'float');
                    fclose(fid);
                    clear fid;
                    figure(1);
                    plot(0:1/1000:2, mGluRmodulation);
                    title('mGluR modulation function');
                    xlabel('mGluR'); ylabel('Ca2+');
                    print([directory,'mGluRmodulation'],'-dpng');
                    clear mGluRmodulation;
                    % Main mGluR
                    fid = fopen([directory,'mGluR',fileExt],'r');
                    XdimInner = fread(fid, 1, 'int');
                    YdimInner = fread(fid, 1, 'int');
                    ZdimInner = fread(fid, 1, 'int');
                    mGluR = fread(fid, (XdimInner*YdimInner*ZdimInner)*(T/sf), 'float');
                    fclose(fid);
                    clear fid;
                    % 4D: time, X, Y, Z
                    mGluR = reshape(mGluR(1:floor(numel(mGluR)/((XdimInner*YdimInner*ZdimInner)))*XdimInner*YdimInner*ZdimInner), ...
                        [XdimInner, YdimInner, ZdimInner, floor(numel(mGluR)/(XdimInner*YdimInner*ZdimInner))]);
                    mGluR = permute(mGluR, [4, 1, 2, 3]);
                end
                %%
                if (postprocess_Ca)
                    fid = fopen([directory,'Ca',fileExt],'r');
                    XdimInner = fread(fid, 1, 'int');
                    YdimInner = fread(fid, 1, 'int');
                    ZdimInner = fread(fid, 1, 'int');
                    Ca = fread(fid, (XdimInner*YdimInner*ZdimInner)*(T/sf), 'float');
                    fclose(fid);
                    clear fid;
                    % 4D: time, X, Y, Z
                    Ca = reshape(Ca(1:floor(numel(Ca)/((XdimInner*YdimInner*ZdimInner)))*XdimInner*YdimInner*ZdimInner), ...
                        [XdimInner, YdimInner, ZdimInner, floor(numel(Ca)/(XdimInner*YdimInner*ZdimInner))]);
                    Ca = permute(Ca, [4, 1, 2, 3]);
                end
                %%
                if (postprocess_ECB)
                    % ECB production function only first
                    fid = fopen([directory,'ECBproduction',fileExt],'r');
                    ECBproduction = fread(fid, Inf, 'float');
                    fclose(fid);
                    clear fid;
                    figure(2);
                    plot(0:1/1000:2, ECBproduction);
                    title('ECB production function');
                    xlabel('Ca2+'); ylabel('ECB');
                    print([directory,'ECBproduction'],'-dpng');
                    % Main ECB
                    fid = fopen([directory,'ECB',fileExt],'r');
                    XdimInner = fread(fid, 1, 'int');
                    YdimInner = fread(fid, 1, 'int');
                    ZdimInner = fread(fid, 1, 'int');
                    ECB = fread(fid, (XdimInner*YdimInner*ZdimInner)*(T/sf), 'float');
                    fclose(fid);
                    clear fid;
                    % 4D: time, X, Y, Z
                    ECB = reshape(ECB(1:floor(numel(ECB)/((XdimInner*YdimInner*ZdimInner)))*XdimInner*YdimInner*ZdimInner), ...
                        [XdimInner, YdimInner, ZdimInner, floor(numel(ECB)/(XdimInner*YdimInner*ZdimInner))]);
                    ECB = permute(ECB, [4, 1, 2, 3]);
                end
                %%
                if (postprocess_OutputSpikes)
                    fid = fopen([directory,'Spike',fileExt],'r');
                    Xdim = fread(fid, 1, 'int');
                    Ydim = fread(fid, 1, 'int');
                    Zdim = fread(fid, 1, 'int');
                    % temp(:,1) is neuron id and temp(:,2) is spike time
                    temp = fread(fid, Inf, 'int'); % have to load all sadly
                    temp = reshape(temp, [2, numel(temp)/2]);
                    fclose(fid);
                    clear fid;
                    if (postprocess_OutputSpikesFilter)
                        temp(:,temp(2,:)>outputSpikeTimesTmax/dt) = []; % filter out greater than Tmin
                        temp(:,temp(2,:)<outputSpikeTimesTmin/dt) = []; % filter out less than Tmax
                    else
                        temp(:,temp(2,:)>T/dt) = []; % filter out greater than T
                    end
                    temp(1,:) = temp(1,:)+1; % because of 0 indexing
                    temp = sortrows(temp'); % Soft by neuron id
                    [~, temp_i] = unique(temp(:,1)); % First spike of each neuron (for those that have spikes)
                    spikesN = circshift(temp_i,-1) - temp_i; % Number of spikes for each neuron (for those that have spikes)
                    spikesN(end) = size(temp,1) - sum(spikesN(1:end-1)); % fix last neuron
                    % {3D cell: X, Y, Z} (1D: spike times)
                    outputSpike = cell(Xdim,Ydim,Zdim).*-1;
                    i=1; % Neuron id counter
                    k=1; % Position in temp_i and spikesN
                    for z=1:Zdim
                        for y=1:Ydim
                            for x=1:Xdim
                                % If this neuron has spikes
                                if (temp(temp_i(k),1) == i)
                                    outputSpike{x,y,z} = ...
                                        temp(temp_i(k):temp_i(k)+spikesN(k)-1,2);
                                    k=k+1;
                                end
                                i=i+1;
                            end
                        end
                    end
                    clear temp temp_i spikesN i k z y x;
                end
                %%
            end
            %% Plot parameters
            sfRange = Tmin+sf:sf:Tmax; % x-axis scaling
            idxRange = (Tmin/sf)+1:Tmax/sf; % x-axis indexing
            if (exist('Xdim','var') && exist('Ydim','var') && exist('Zdim','var') && ...
                    exist('XdimInner','var') && exist('YdimInner','var') && exist('ZdimInner','var'))
                Xmin=1; Xmax=Xdim;
                Ymin=1; Ymax=Ydim;
                Zmin=1; Zmax=Zdim;
                XminInner=1; XmaxInner=XdimInner;
                YminInner=1; YmaxInner=YdimInner;
                ZminInner=1; ZmaxInner=ZdimInner;
            else
                warning('Need to load data to plot.');
                return;
            end
            %% Plot
            if (postprocess_InputSpikes)
                figure(3); % Spikes
                numInputSpikes=0;
                for i=1:Xdim
                    numInputSpikes = numInputSpikes + numel(inputSpike{i,1,1});
                end
                temp = zeros(numInputSpikes,2);
                j = 1;
                for i=1:Xdim % only plot Xdim
                    n = numel(inputSpike{i,1,1});
                    temp(j:j+n-1,1) = ...
                        ones(n,1)*i;
                    temp(j:j+n-1,2) = inputSpike{i,1,1};
                    j = j + n;
                end
                clear j i n;
                scatter(temp(:,2).*dt,temp(:,1));
                title('Spiking activity (X-dimension only)');
                xlim([Tmin Tmax]);
                print([directory,'inSpikes'],'-dpng');
                clear i temp numInputSpikes;
                figure(4);% Population firing rate histogram over all time
                i=1;
                inHz=zeros(Xdim*Ydim*Zdim,numel(0:HzSf:T-HzSf));
                for x=1:Xdim
                    for y=1:Ydim
                        for z=1:Zdim
                            for t=0:HzSf:T-HzSf
                               inHz(i,t+1) = ...
                                    numel(find(...
                                        inputSpike{x,y,z}>=t/dt & ...
                                        inputSpike{x,y,z}<(t+1)/dt...
                                    ))*(1.0/HzSf);
                            end
                            i=i+1;
                        end
                    end
                end
                clear i x y z t;
                if (postprocess_Perturbation)
                    histogram(mean(inHz(:,1:floor(perturbationT/HzSf)-1),2));
                    title('In Hz before perturbation');
                else
                    histogram(mean(inHz,2));
                    title('In Hz');
                end
                title('Firing rates');
                print([directory,'inHz'],'-dpng');
            end
            %%
            synapse=randi(XdimInner,1,1); % only for Xdimension
            %%
            if (postprocess_InputSpikes)
                figure(5);
                subplot(9,1,1); 
                scatter(inputSpike{preIndexs(1,synapse,1,1),1,1}.*dt, ...
                    ones(numel(inputSpike{preIndexs(1,synapse,1,1),1,1}),1));
                set(gca,'XTick',[]);
                clear inputSpike;
            end
            %%
            if (postprocess_Glutamate)
                figure(5);
                subplot(9,1,2); hold on;
                plot(sfRange, glutamate(idxRange,synapse,1,1));
                xlim([Tmin Tmax]);
                set(gca,'XTick',[]);
                clear glutamate;
            end
            %%
            if (postprocess_AvailableGlutamate)
                figure(5);
                subplot(9,1,2); hold on;
                plot(sfRange, availableGlutamate(idxRange,synapse,1,1));
                xlim([Tmin Tmax]);
                set(gca,'XTick',[]);
            end
            %%
            if (postprocess_Cb1R)
                figure(5);
                subplot(9,1,3);
                plot(sfRange, Cb1R(idxRange,synapse,1,1));
                xlim([Tmin Tmax]);
                set(gca,'XTick',[]);
                clear Cb1R;
            end 
            %%
            if (postprocess_CleftGlutamate)
                figure(5);
                subplot(9,1,4);
                plot(sfRange, cleftGlutamate(idxRange,synapse,1,1));
                xlim([Tmin Tmax]);
                set(gca,'XTick',[]);
            end
            %%
            if (postprocess_AMPA || postprocess_AMPAWeights)
                figure(5);
                subplot(9,1,5); hold on;
                if (postprocess_AMPAWeights)
                    plot(sfRange, ones(1,numel(sfRange))*AMPAWeights(synapse,1,1));
                    yyaxis right;
                end
                if (postprocess_AMPA)
                    plot(sfRange, AMPA(idxRange,synapse,1,1));  
                end
                xlim([Tmin Tmax]);
                set(gca,'XTick',[]);
                clear AMPA;
            end
            %%
            if (postprocess_mGluR)
                figure(5);
                subplot(9,1,6);
                plot(sfRange, mGluR(idxRange,synapse,1,1));
                xlim([Tmin Tmax]);
                set(gca,'XTick',[]);
            end
            %%
            if (postprocess_Ca)
                figure(5);
                subplot(9,1,7);
                plot(sfRange, Ca(idxRange,synapse,1,1));
                xlim([Tmin Tmax]);
                set(gca,'XTick',[]);
            end
            %%
            if (postprocess_ECB)
                figure(5);
                subplot(9,1,8);
                plot(sfRange, ECB(idxRange,synapse,1,1));
                xlim([Tmin Tmax]);
                set(gca,'XTick',[]);
            end
            %%
            if (postprocess_OutputSpikes)
                figure(5);
                subplot(9,1,9);
                scatter(outputSpike{preIndexs(1,synapse,1,1),1,1}.*dt, ...
                    ones(numel(outputSpike{preIndexs(1,synapse,1,1),1,1}),1));
                set(gca,'XTick',[]);
            end
            clear preIndexs;
            %%
            if (postprocess_InputSpikes || postprocess_Glutamate ...
                    || postprocess_Cb1R || postprocess_CleftGlutamate ...
                    || postprocess_AMPA || postprocess_AMPAWeights ...
                    || postprocess_mGluR || postprocess_Ca ...
                    || postprocess_ECB || postprocess_OutputSpikes)
                print([directory,'components'],'-dpng');
            end   
            %%
            if (postprocess_mGluR && postprocess_Ca && postprocess_ECB)
                figure(6);
                Hrange = 0:3/50:3;
                mGluRHist = zeros(numel(Hrange),size(mGluR,1));
                for i=1:size(mGluR,1)
                    tempmGluR = mGluR(i,:,:,:);
                    mGluRHist(:,i) = hist(tempmGluR(:),Hrange);
                end
                subplot(3,1,1);
                imagesc([0 size(mGluR,1)],Hrange,mGluRHist);
                title('mGluR');
                colormap(flipud(hot)); colorbar(); caxis([0 cbMax]);
                set(gca,'ydir','normal');
                Hrange = 0:3/50:3;
                CaHist = zeros(numel(Hrange),size(Ca,1));
                for i=1:size(Ca,1)
                    tempCa = Ca(i,:,:,:);
                    CaHist(:,i) = hist(tempCa(:),Hrange);
                end
                subplot(3,1,2);
                imagesc([0 size(Ca,1)],Hrange,CaHist);
                title('Ca2+');
                colormap(flipud(hot)); colorbar(); caxis([0 cbMax]);
                set(gca,'ydir','normal');
                Hrange = 0:1/50:1;
                ECBHist = zeros(numel(Hrange),size(ECB,1));
                for i=1:size(ECB,1)
                    tempECB = ECB(i,:,:,:);
                    ECBHist(:,i) = hist(tempECB(:),Hrange);
                end
                subplot(3,1,3);
                imagesc([0 size(ECB,1)],Hrange,ECBHist);
                title('ECB');
                colormap(flipud(hot)); colorbar(); caxis([0 cbMax]);
                set(gca,'ydir','normal');
                print([directory,'mGluR_Ca_ECB'],'-dpng');
                clear Hrange i mGluRHist tempmGluR CaHist tempCa ECBHist tempECB;
                clear mGluR Ca ECB;
            end 
            %%
            if (postprocess_OutputSpikes)
                figure(7); % Spikes
                numOutputSpikes=0;
                for i=1:Xdim
                    numOutputSpikes = numOutputSpikes + numel(outputSpike{i,1,1});
                end
                temp = zeros(numOutputSpikes,2);
                j = 1;
                for i=1:Xdim % only plot Xdim
                    n = numel(outputSpike{i,1,1});
                    temp(j:j+n-1,1) = ...
                        ones(n,1)*i;
                    temp(j:j+n-1,2) = outputSpike{i,1,1};
                    j = j + n;
                end
                clear j i n;
                scatter(temp(:,2).*dt,temp(:,1));
                title('Spiking activity (X-dimension only)');
                xlim([Tmin Tmax]);
                print([directory,'outSpikes'],'-dpng');
                clear i temp numOutputSpikes;
                figure(8);% Population firing rate histogram over all time
                i=1;
                outHz=zeros(Xdim*Ydim*Zdim,numel(0:HzSf:T-HzSf));
                for x=1:Xdim
                    for y=1:Ydim
                        for z=1:Zdim
                            for t=0:HzSf:T-HzSf
                               outHz(i,t+1) = ...
                                    numel(find(...
                                        outputSpike{x,y,z}>=t/dt & ...
                                        outputSpike{x,y,z}<(t+1)/dt...
                                    ))*(1.0/HzSf);
                            end
                            i=i+1;
                        end
                    end
                end
                clear i x y z t;
                if (postprocess_Perturbation)
                    histogram(mean(outHz(:,1:floor(perturbationT/HzSf)-1),2));
                    title('Out Hz before perturbation');
                else
                    histogram(mean(outHz,2));
                    title('Out Hz');
                end
                title('Firing rates');
                print([directory,'outHz'],'-dpng');
            end
            %%
            if (postprocess_Means)
                Hrange = 0:0.01:2.1;
                availableGlutamateHist = zeros(numel(Hrange),size(availableGlutamate,1));
                for i=1:size(availableGlutamate,1)
                    tempAvailableGlutamate = availableGlutamate(i,:,:,:);
                    availableGlutamateHist(:,i) = hist(tempAvailableGlutamate(:),Hrange);
                end
                figure(9);
                imagesc([0 size(availableGlutamate,1)],Hrange,availableGlutamateHist);
                title('Available Glutamate ');
                colormap(flipud(hot)); colorbar(); caxis([0 cbMax/2]);
                set(gca,'ydir','normal');
                print([directory,'availableGlutamate_hist'],'-dpng');
                clear i tempAvailableGlutamate availableGlutamateHist;
            end
            %%
            if (postprocess_Headroom)
                Hrange = -2.1:0.01:2.1;
                figure(10);
                headroomH = zeros(numel(Hrange),size(availableGlutamate,1));  
                for i=1:size(availableGlutamate,1) % over time, glutamate and AMPA are same size
                    % calculate the diff/headroom for each synapse and hist it
                    tempAvailableGlutamate = availableGlutamate(i,:,:,:);
                    headroomH(:,i) = hist(tempAvailableGlutamate(:)-AMPAWeights1D, ...
                        Hrange);
                end
                imagesc([0 size(availableGlutamate,1)],Hrange,headroomH);
                colormap(flipud(hot)); colorbar();
                set(gca,'ydir','normal');
                print([directory,'headroom'],'-dpng');
                figure(11);
                if (postprocess_Perturbation)
                    subplot(2,1,1);
                    plot(Hrange,headroomH(:,(perturbationT/sf)-1));
                    xlim([Hrange(1) Hrange(end)]);
                    title('Before perturbation');
                    subplot(2,1,2);
                    plot(Hrange,headroomH(:,end));
                    xlim([Hrange(1) Hrange(end)]);
                    title('After perturbation');
                else
                    plot(Hrange,headroomH(:,end));
                    xlim([Hrange(1) Hrange(end)]);
                end
                print([directory,'headroomDist'],'-dpng');
                clear i Hrange tempAvailableGlutamate headroomH;
                %%
                figure(12); % correlation of AMPA and headroom
                headroomFinal = zeros(1,XdimInner*YdimInner*ZdimInner);
                i = 1;
                for x=1:XdimInner
                    for y=1:YdimInner
                        for z=1:ZdimInner
                            tempAvailableGlutamate = availableGlutamate(:,x,y,z);
                            headroomFinal(i) = mean(tempAvailableGlutamate( ...
                                finalHeadroomT(1)/sf:finalHeadroomT(end)/sf) - ...
                                AMPAWeights(x,y,z));
                            i = i + 1;
                        end
                    end
                end
                scatter(AMPAWeights1D, headroomFinal);
                h=lsline;
                set(h,'color','r');
                R=corrcoef(AMPAWeights1D, headroomFinal);
                R_squared=R(2)^2;
                title(['AMPA vs headroom ', '(R squared= ', num2str(R_squared), ')']);
                print([directory,'AMPAvsHeadroom'],'-dpng');
                clear i tempAvailableGlutamate tempAMPA h R R_squared;
                %%
                figure(13); % correlation of input Hz and headroom
                HzFinal = zeros(size(inHz,1),1);
                for i=1:size(inHz,1) % over pre-synaptic neurons, what is mean end Hz
                    HzFinal(i) = mean(inHz(i,finalHzT(1)/HzSf: ...
                        finalHzT(end)/HzSf));
                end
                scatter(HzFinal(preIndexs1D), headroomFinal);
                xlim([0 HzMax]);
                h=lsline;
                set(h,'color','r');
                R=corrcoef(HzFinal(preIndexs1D), headroomFinal);
                R_squared=R(2)^2;
                title(['Input Hz vs headroom ', '(R squared= ', num2str(R_squared), ')']);
                print([directory,'HzvsHeadroom'],'-dpng');
                clear i tempPreIndexs R R_squared;
                %%
                figure(14); % correlation of total effect of Hz and AMPA and headroom
                scatter(HzFinal(preIndexs1D).*AMPAWeights1D, headroomFinal);
                h=lsline;
                set(h,'color','r');
                R=corrcoef(HzFinal(preIndexs1D).*AMPAWeights1D, headroomFinal);
                R_squared=R(2)^2;
                title(['(Input Hz * AMPA) vs headroom ', '(R squared= ', num2str(R_squared), ')']);
                print([directory,'HzAMPAvsHeadroom'],'-dpng');
                clear R R_squared;
                %%
                % Time to headroom
                timeToHeadroom = zeros(1,XdimInner*YdimInner*ZdimInner);
                i = 1;
                for x=1:XdimInner
                    for y=1:YdimInner
                        for z=1:ZdimInner
                            tempAvailableGlutamate = availableGlutamate(:,x,y,z);
                            tempHeadroom = tempAvailableGlutamate - AMPAWeights(x,y,z);
                            temp = find(tempHeadroom<=headroomFinal(i));
                            timeToHeadroom(i) = temp(1);
                            i = i + 1;
                        end
                    end
                end
                %%
                figure(15); % AMPA vs time to headroom
                scatter(AMPAWeights1D, timeToHeadroom);
                xlim([0 AMPAMax]);
                h=lsline;
                set(h,'color','r');
                R=corrcoef(AMPAWeights1D, timeToHeadroom);
                R_squared=R(2)^2;
                title(['AMPA vs time to headroom ', '(R squared= ', num2str(R_squared), ')']);
                print([directory,'AMAPvsHeadroomTime'],'-dpng');
                clear i tempAvailableGlutamate tempHeadroom temp R R_squared;
                clear availableGlutamate AMPAWeights;
                %%
                figure(16); % input Hz vs time to headroom
                scatter(HzFinal(preIndexs1D), timeToHeadroom);
                xlim([0 HzMax]);
                h=lsline;
                set(h,'color','r');
                R=corrcoef(HzFinal(preIndexs1D), timeToHeadroom);
                R_squared=R(2)^2;
                title(['Input Hz vs time to headroom ', '(R squared= ', num2str(R_squared), ')']);
                print([directory,'HzvsHeadroomTime'],'-dpng');
                %%
                figure(17); % Hz and AMPA vs time to headroom
                scatter(HzFinal(preIndexs1D).*AMPAWeights1D, timeToHeadroom);
                h=lsline;
                set(h,'color','r');
                R=corrcoef(HzFinal(preIndexs1D).*AMPAWeights1D, timeToHeadroom);
                R_squared=R(2)^2;
                title(['(Input Hz * AMPA) vs time to headroom ', '(R squared= ', num2str(R_squared), ')']);
                print([directory,'HzAMPAvsHeadroomTime'],'-dpng');
                clear tempGlutamate tempAMPA tempHeadroom temp temp2;
                %%
                if (postprocess_RiskySynapses)
                    % which synapses are "risky"
                    riskyGroup = zeros(1,numel(preIndexs1D));
                    riskyGroup(headroomFinal<headroomRisk ...
                        & timeToHeadroom>timeToHeadroomRisk) = 1;
                    riskyGroup(headroomFinal>headroomRisk ...
                        & timeToHeadroom>timeToHeadroomRisk) = 2;
                    riskyGroup(headroomFinal>headroomRisk ...
                        & timeToHeadroom<timeToHeadroomRisk) = 3; 
                    %%
                    figure(18); clf;
                    scatterhist(headroomFinal, timeToHeadroom, ...
                        'Kernel', 'on');
                    xlabel('Headroom'); ylabel('Time to headroom');
                    print([directory,'headroomVsTimeToHeadroom'],'-dpng');
                    legend off;
                    %%
                    figure(19);
                    hp1 = uipanel('position', [0.0 0.5 0.5 0.5]);
                    scatterhist(headroomFinal, timeToHeadroom, ...
                        'Group', riskyGroup, 'Kernel', 'on', 'Parent', hp1);
                    xlabel('Headroom'); ylabel('Time to headroom');
%                     legend off;

                    hp2 = uipanel('position', [0.5 0.5 0.5 0.5]);
                    tempHz = HzFinal(preIndexs1D);
                    scatterhist(AMPAWeights1D, tempHz, ...
                        'Group', riskyGroup, 'Kernel', 'on', 'Parent', hp2);
                    xlim([0 AMPAMax]); xlabel('Glutamate bound to AMPA');
                    ylim([0 HzMax]); ylabel('In Hz');
                    legend off;

                    hp3 = uipanel('position', [0.0 0.0 0.5 0.5]);
                    tempCleft = sum(cleftGlutamate(finalCleftGlutamateT(1)/sf: ...
                        finalCleftGlutamateT(end)/sf,:,:,:));
                    scatterhist(AMPAWeights1D, tempCleft(:), ...
                        'Group', riskyGroup, 'Kernel', 'on', 'Parent', hp3);
                    xlim([0 AMPAMax]); xlabel('Glutamate bound to AMPA');
                    ylabel('Glutamate flux');
                    legend off;

                    hp4 = uipanel('position', [0.5 0.0 0.5 0.5]);
                    scatterhist(tempHz, tempCleft(:), ...
                        'Group', riskyGroup, 'Kernel', 'on', 'Parent', hp4);
                    xlim([0 HzMax]); xlabel('In Hz');
                    ylabel('Glutamate flux');
                    legend off;

                    print([directory,'riskySynapses'],'-dpng');
                    clear tempHz tempCleft;
                    clear preIndexs1D cleftGlutamate AMPAWeights1D ...
                        headroomFinal HzFinal timeToHeadroom;
                    %%
                    if (postprocess_Perturbation)
                        %%
                        if (postprocess_PerturbationHz)
                            % Work out before and after Hz
                            beforePerturbationHz = ...
                                mean(inHz(:,1:(perturbationT/HzSf)-1),2);
                            afterPerturbationHz ...
                                = mean(inHz(:,perturbationT/HzSf:end),2);
                            diffPerturbationHz = afterPerturbationHz - ...
                                beforePerturbationHz;
                            % Work out headroom factors
                            
                            % Plot scatters of the perturbation and factors
%                             figure(20);
%                             scatter(diffPerturbationHz(preIndexs1D), headroom);
%                             scatter(diffPerturbationHz(preIndexs1D), timeToHeadroom);
%                             scatter(diffPerturbationHz(preIndexs1D), glutamateFluxPerturbation);
                            clear inHz;
                        end
                        %%
                        % Find those that were below 30 before the perturbation
                        % period.
                        lowHz=find(mean(inHz(:,1:perturbationT-1),2) <= 30.0);
                        % Work out the time when they re-converged
%                         tempToHeadroomPerturbation = zeros(1,size(glutamate,2));
%                         for i=1:size(glutamate,2)
%                             tempGlutamate = glutamate(perturbationT/sf:end,i,:,:,2);
%                             tempAMPA = AMPA(perturbationT/sf:end,i,:,:,1);
%                             tempHeadroom = tempGlutamate(:) - tempAMPA(:);
%                             temp = find(tempHeadroom<=headroomFinal(i));
%                             tempToHeadroomPerturbation(i) = temp(1);
%                         end
                        % Work out their excess 'wasteful' glutamate
%                         tempCleftGlutamate = cleftGlutamate;
%                         tempAMPA = AMPA(:,:,1,1,1);
                        waste = tempCleftGlutamate;% - tempAMPA;
    %                     waste(waste<0) = 0;
                        totalWaste = zeros(1,Xdim);
                        for i=1:Xdim
                           totalWaste(i) = sum(waste(...
                               perturbationT/sf:(perturbationT/sf)+...
                               tempToHeadroomPerturbation(i),...
                               i));
    %                        totalWaste(i) = sum(waste(...
    %                            1:end,...
    %                            i));
                        end
                        subplot(2,2,1);
                        scatter(tempAMPA(end,lowHz),tempToHeadroomPerturbation(lowHz))
                        subplot(2,2,2);
                        scatter(tempToHeadroomPerturbation(lowHz),totalWaste(lowHz))
                        subplot(2,2,3);
                        scatter(tempAMPA(end,lowHz),totalWaste(lowHz));
                        subplot(2,2,2);
                        scatter(mean(inHz(lowHz,1:perturbationT-1),2), ...
                            totalWaste(lowHz))
                        subplot(2,2,3);
                        lowAMPA = find(tempAMPA(end,:) < 0.15);
                        highAMPA = find(tempAMPA(end,:) >= 0.15);
                        nonZeroTotalWaste = find(totalWaste>000);
                        lowNonZero = intersect(lowAMPA,nonZeroTotalWaste);
                        highNonZero = intersect(highAMPA,nonZeroTotalWaste);
                        bar([1,2],...
                            [mean(totalWaste(intersect(lowHz,lowNonZero))),...
                            mean(totalWaste(intersect(lowHz,highNonZero)))]);
                        subplot(2,2,4);
                        scatter(tempAMPA(end,:),tempToHeadroomPerturbation);
                        clear tempCleftGlutamate tempAMPA;
                    end
                end
                %%
                if (parameterSearch)
%                     temp = glutamate(end-9:end,:,:,:,2) - ...
%                         lAMPA(end-9:end,:,:,:,1);
%                     headroomMean(vX_r==vX_range, vY_r==vY_range, t_r==t_range) = ...
%                         mean(temp(:));
%                     headroomRange(vX_r==vX_range, vY_r==vY_range, t_r==t_range) = ...
%                         range(temp(:));
%                     headroomSum(vX_r==vX_range, vY_r==vY_range, t_r==t_range) = ...
%                         sum(temp(:));
%                     clear temp;
                end
            end
            %%
        end
    end
end
clear vX_r vY_r t_r;
%% Plot headroom as a function of the parameter search
if (postprocess_Headroom && parameterSearch)
    figure(21); clf; 
    imagesc(flipud(headroomMean)); colorbar();
    figure(22); clf; 
    imagesc(flipud(headroomRange)); colorbar();
    figure(23); clf; 
    imagesc(flipud(headroomSum)); colorbar();
end
