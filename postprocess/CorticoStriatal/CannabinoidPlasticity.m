<<<<<<< HEAD
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
=======
clear variables; %close all;
set(0,'defaulttextinterpreter','latex'); rng('shuffle');
%% Parameters
figNum=0;
% general parameters
dt=0.0001;
% sf=dt; % sampling interval in s
% sf=0.0010;
sf=0.0020; % typical
% sf=0.0100;
HzSf=1; % window (and step) to calculate Hz in s
HzMax=160; % limit for input Hz axes
cbMax=100; % scaling of color bar for imagesc plots
AMPAMax=1.5; % limit for AMPA axes
directory='../../graphs/CorticoStriatal/';
% directory='../../graphs/CorticoStriatal/HzPerturbation/';
% directory='../../graphs/CorticoStriatal/AMPAPerturbation/';
% directory='/home/jmhumble/Data/cannabinoids/00004_e1b77fc_XXX/bothPerturbation/9/';
fileExt='.dat';
% time frames
T=0:499; % Length of simulation time saving data (excluding spikes) in s to load and process
Tperturbation=500:1000; % same as above but for a perturbation
postprocess_InputSpikesFilter=true; % whether to filter input spikes
inputSpikeTimesT=449:T(end); % time window to filter
inputSpikeTimesTperturbation=949:Tperturbation(end); % same as above but for a perturbation
postprocess_OutputSpikesFilter=true; % whether to filter output spikes
outputSpikeTimesT=inputSpikeTimesT; % time window to filter
outputSpikeTimesTperturbation=inputSpikeTimesTperturbation; % same as above but for a perturbation
finalMeasurementT=inputSpikeTimesT; % time window to calculate final measurements
finalMeasurementTperturbation=inputSpikeTimesTperturbation; % same as above but for a perturbation
% additional parameters
finalHeadroomStd=1; % accuracy of final headroom for determining time to headroom
headroomRisk=0.4; % the headroom demarcation line for risky synapses
%% Whether to load/process different data sources
postprocess_PreIndexs = true;
postprocess_InputSpikes = true;
>>>>>>> origin/team-A
postprocess_Glutamate = false;
postprocess_AvailableGlutamate = true;
postprocess_Cb1R = false;
postprocess_CleftGlutamate = true;
<<<<<<< HEAD
=======
postprocess_CleftECB = false;
>>>>>>> origin/team-A
postprocess_AMPA = false;
postprocess_AMPAWeights = true;
postprocess_mGluR = false;
postprocess_Ca = false;
postprocess_ECB = false;
postprocess_OutputSpikes = false;
<<<<<<< HEAD
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
=======
postprocess_Headroom = true;
postprocess_RiskySynapses = true;
postprocess_Perturbation = true;
postprocess_PerturbationHz = true;
postprocess_PerturbationAMPA = true;
%%
for perturbation=0:postprocess_Perturbation
    clear inputSpike glutamate availableGlutamate preIndexs preIndexs1D ...
        Cb1R cleftGlutamate AMPA AMPAWeights AMPAWeights1D mGluR Ca ECB ...
        outputSpike;
    % Change variables if a perturbation
    if (~perturbation)
        Trange=T;
        spikeTrange=inputSpikeTimesT;
        measurementTrange=finalMeasurementT;
    else
        Trange=Tperturbation;
        spikeTrange=inputSpikeTimesTperturbation;
        measurementTrange=finalMeasurementTperturbation;
    end
    sfRange = Trange(1)+sf:sf:Trange(end);
    % Load data
    if (postprocess_InputSpikes)
        [inputSpike, Xdim, Ydim, Zdim] = loadSpikes(directory, ...
            'PoissonSpikes', fileExt, postprocess_InputSpikesFilter, ...
            spikeTrange(1), spikeTrange(end), Trange(1), Trange(end), dt);
    end
    if (postprocess_Glutamate)
        [glutamate, XdimInner, YdimInner, ZdimInner] = ...
            load4D(directory, 'Glutamate', fileExt, Trange(1), ...
            Trange(end), sf);
    end 
    if (postprocess_AvailableGlutamate)
        [availableGlutamate, XdimInner, YdimInner, ZdimInner] = ...
            load4D(directory, 'AvailableGlutamate', fileExt, Trange(1), ...
            Trange(end), sf);
    end 
    if (postprocess_PreIndexs)
        [preIndexs, preIndexs1D] = load2D(directory, 'Indexs', ...
            fileExt, XdimInner, YdimInner, ZdimInner);
    end
    if (postprocess_Cb1R)
        [Cb1R, XdimInner, YdimInner, ZdimInner] = ...
            load4D(directory, 'Cb1R', fileExt, Trange(1), ...
            Trange(end), sf);
    end 
    if (postprocess_CleftGlutamate)
        [cleftGlutamate, XdimInner, YdimInner, ZdimInner] = ...
            load4D(directory, 'CleftGlutamate', fileExt, Trange(1), ...
            Trange(end), sf);
    end 
    if (postprocess_CleftECB)
        [cleftECB, XdimInner, YdimInner, ZdimInner] = ...
            load4D(directory, 'CleftECB', fileExt, Trange(1), ...
            Trange(end), sf);
    end     
    if (postprocess_AMPA)
        [AMPA, XdimInner, YdimInner, ZdimInner] = ...
            load4D(directory, 'AMPA', fileExt, Trange(1), ...
            Trange(end), sf);
    end
    if (postprocess_AMPAWeights)
        if (perturbation && postprocess_PerturbationAMPA)
            [AMPAWeights, AMPAWeights1D] = load3D(directory, ...
                ['AMPAWeights_',num2str(Tperturbation(1)/dt)], fileExt, ...
                XdimInner, YdimInner, ZdimInner);
        else
            [AMPAWeights, AMPAWeights1D] = load3D(directory, ...
                'AMPAWeights_1', fileExt, XdimInner, YdimInner, ...
                ZdimInner);
        end
    end
    if (postprocess_mGluR)
        if (~perturbation)
            [~, figNum] = newFigure(figNum, false);
            plotModulation(directory, 'mGluRmodulation', fileExt, ...
                'mGluR modulation function', 'mGluR', 'Ca2+', ...
                'mGluRmodulation');
        end
        [mGluR, XdimInner, YdimInner, ZdimInner] = ...
            load4D(directory, 'mGluR', fileExt, Trange(1), ...
            Trange(end), sf);
    end
    if (postprocess_Ca)
        [Ca, XdimInner, YdimInner, ZdimInner] = ...
            load4D(directory, 'Ca', fileExt, Trange(1), ...
            Trange(end), sf);
    end
    if (postprocess_ECB)
        if (~perturbation)
            [~, figNum] = newFigure(figNum, false);
            plotModulation(directory, 'ECBproduction', fileExt, ...
                'ECB modulation function', 'Ca2+', 'ECB+', ...
                'ECBproduction');
        end
        [ECB, XdimInner, YdimInner, ZdimInner] = ...
            load4D(directory, 'ECB', fileExt, Trange(1), ...
            Trange(end), sf);
    end
    if (postprocess_OutputSpikes)
        [outputSpike, Xdim, Ydim, Zdim] = loadSpikes(directory, ...
            'Spike', fileExt, postprocess_OutputSpikesFilter, ...
            spikeTrange(1), spikeTrange(end), Trange(1), Trange(end), dt);
    end
    % Plot
    if (postprocess_InputSpikes)
        [~, figNum] = newFigure(figNum, false); % Population firing rate histogram over all time
        inHz = calculateHz(inputSpike, spikeTrange(1), ...
            spikeTrange(end), HzSf, Xdim, ...
            Ydim, Zdim, dt);
        histogram(mean(inHz(:),2));
        title(['In Hz',perturbationString(perturbation,0,1)]); 
        xlabel('Hz'); ylabel('count');
        print([directory,'inHz',perturbationString(perturbation,1,0)],'-dpng');
    end
    synapse=randi(XdimInner,1,1); % only for Xdimension
    if (postprocess_InputSpikes)
        [~, figNum] = newFigure(figNum, false);
        subplot(10,1,1); 
        scatter(inputSpike{preIndexs(1,synapse,1,1),1,1}.*dt, ...
            ones(numel(inputSpike{preIndexs(1,synapse,1,1),1,1}),1),'.');
        title(['Components',perturbationString(perturbation,0,1)]); 
        xlim([Trange(1) Trange(end)]);
        set(gca,'XTick',[]);
    end
    if (postprocess_Glutamate || postprocess_AvailableGlutamate)
        subplot(10,1,2); hold on;
        if (postprocess_Glutamate)
            plot(sfRange, glutamate(:,synapse,1,1));
        end
        if (postprocess_AvailableGlutamate)
            plot(sfRange, availableGlutamate(:,synapse,1,1));
        end
        xlim([Trange(1) Trange(end)]);
        set(gca,'XTick',[]);
    end
    if (postprocess_Cb1R)
        subplot(10,1,3);
        plot(sfRange, Cb1R(:,synapse,1,1));
        xlim([Trange(1) Trange(end)]);
        set(gca,'XTick',[]);
    end 
    if (postprocess_CleftGlutamate)
        subplot(10,1,4);
        plot(sfRange, cleftGlutamate(:,synapse,1,1));
        xlim([Trange(1) Trange(end)]);
        set(gca,'XTick',[]);
    end
    if (postprocess_CleftECB)
        subplot(10,1,5);
        plot(sfRange, cleftECB(:,synapse,1,1));
        xlim([Trange(1) Trange(end)]);
        set(gca,'XTick',[]);
    end    
    if (postprocess_AMPA || postprocess_AMPAWeights)
        subplot(10,1,6); hold on;
        if (postprocess_AMPAWeights)
            plot(sfRange, ones(1,numel(sfRange))*AMPAWeights(synapse,1,1));
            yyaxis right;
        end
        if (postprocess_AMPA)
            plot(sfRange, AMPA(:,synapse,1,1));  
        end
        xlim([Trange(1) Trange(end)]);
        set(gca,'XTick',[]);
    end
    if (postprocess_mGluR)
        subplot(10,1,7);
        plot(sfRange, mGluR(:,synapse,1,1));
        xlim([Trange(1) Trange(end)]);
        set(gca,'XTick',[]);
    end
    if (postprocess_Ca)
        subplot(10,1,8);
        plot(sfRange, Ca(:,synapse,1,1));
        xlim([Trange(1) Trange(end)]);
        set(gca,'XTick',[]);
    end
    if (postprocess_ECB)
        subplot(10,1,9);
        plot(sfRange, ECB(:,synapse,1,1));
        xlim([Trange(1) Trange(end)]);
        set(gca,'XTick',[]);
    end
    if (postprocess_OutputSpikes)
        subplot(10,1,10);
        scatter(outputSpike{preIndexs(1,synapse,1,1),1,1}.*dt, ...
            ones(numel(outputSpike{preIndexs(1,synapse,1,1),1,1}),1),'.');
        xlim([Trange(1) Trange(end)]);
        set(gca,'XTick',[]);
    end
    if (postprocess_InputSpikes || postprocess_Glutamate ...
            || postprocess_AvailableGlutamate || postprocess_Cb1R ...
            || postprocess_CleftGlutamate || postprocess_AMPA ...
            || postprocess_AMPAWeights || postprocess_mGluR ...
            || postprocess_Ca || postprocess_ECB ...
            || postprocess_OutputSpikes)
        print([directory,'components',perturbationString(perturbation,1,0)],'-dpng');
    end  
    if (postprocess_OutputSpikes)
        [~, figNum] = newFigure(figNum, false); % Population firing rate histogram over all time
        outHz = calculateHz(outputSpike, spikeTrange(1), ...
            spikeTrange(end), HzSf, Xdim, ...
            Ydim, Zdim, dt);
        histogram(mean(outHz(:),2));
        title(['Out Hz',perturbationString(perturbation,0,1)]); 
        xlabel('Hz'); ylabel('count');
        print([directory,'outHz',perturbationString(perturbation,1,0)],'-dpng');
    end
    if (postprocess_mGluR && postprocess_Ca && postprocess_ECB)
        [~, figNum] = newFigure(figNum, false);
        Hrange = 0:3/50:3;

        mGluRHist = hist3D(Hrange, mGluR);
        subplot(3,1,1);
        imagesc([0 size(mGluR,1)],Hrange,mGluRHist);
        title(['mGluR',perturbationString(perturbation,0,1)]); 
        xlabel('time'); ylabel('mGluR');
        colormap(flipud(hot)); colorbar(); caxis([0 cbMax]);
        set(gca,'ydir','normal');

        CaHist = hist3D(Hrange, Ca);
        subplot(3,1,2);
        imagesc([0 size(Ca,1)],Hrange,CaHist);
        title(['Ca2+',perturbationString(perturbation,0,1)]);
        xlabel('time'); ylabel('Ca2+');
        colormap(flipud(hot)); colorbar(); caxis([0 cbMax]);
        set(gca,'ydir','normal');

        ECBHist = hist3D(Hrange, ECB);
        subplot(3,1,3);
        imagesc([0 size(ECB,1)],Hrange,ECBHist);
        title(['ECB',perturbationString(perturbation,0,1)]);
        xlabel('time'); ylabel('ECB');
        colormap(flipud(hot)); colorbar(); caxis([0 cbMax]);
        set(gca,'ydir','normal');
        print([directory,'mGluR_Ca_ECB',perturbationString(perturbation,1,0)],'-dpng');
        clear Hrange mGluRHist CaHist ECBHist;
    end 
    if (postprocess_AvailableGlutamate)
        Hrange = 0:0.01:2.1;
        availableGlutamateHist = hist3D(Hrange, availableGlutamate);
        [~, figNum] = newFigure(figNum, false);
        imagesc([0 size(availableGlutamate,1)],Hrange,availableGlutamateHist);
        title(['Available Glutamate',perturbationString(perturbation,0,1)]);
        xlabel('time'); 
        ylabel('Available Glutamate');
        colormap(flipud(hot)); colorbar(); caxis([0 cbMax/2]);
        set(gca,'ydir','normal');
        print([directory,'availableGlutamate_hist',perturbationString(perturbation,1,0)],'-dpng');
        clear Hrange availableGlutamateHist;
    end
    if (postprocess_Headroom)
        Hrange = -2.1:0.01:2.1;
        [~, figNum] = newFigure(figNum, false);
        headroomH = histHeadroom3D(Hrange, availableGlutamate, ...
            AMPAWeights1D);
        imagesc([0 size(availableGlutamate,1)],Hrange,headroomH);
        title(['Headroom',perturbationString(perturbation,0,1)]);
        xlabel('time'); ylabel('headroom');
        colormap(flipud(hot)); colorbar();
        set(gca,'ydir','normal');
        print([directory,'headroom',perturbationString(perturbation,1,0)],'-dpng');
        [~, figNum] = newFigure(figNum, false);
        plot(Hrange,headroomH(:,1));
        title(['Headroom distribution - initial condition',perturbationString(perturbation,0,1)]);
        xlabel('headroom'); ylabel('count');
        xlim([Hrange(1) Hrange(end)]);
        print([directory,'headroomDistInitialCondition',perturbationString(perturbation,1,0)],'-dpng');
        [~, figNum] = newFigure(figNum, false);
        plot(Hrange,headroomH(:,end));
        title(['Headroom distribution - steady state',perturbationString(perturbation,0,1)]);
        xlabel('headroom'); ylabel('count');
        xlim([Hrange(1) Hrange(end)]);
        print([directory,'headroomDistSteadyState',perturbationString(perturbation,1,0)],'-dpng');
        clear Hrange headroomH;
        % correlation of AMPA/Hz with headroom/timeToHeadroom
        [headroomFinal, headroomFinalStd, HzFinal, timeToHeadroom, ...
            timeToHeadroomConverged, figNum] = ...
            plotComponentVsHeadroom(figNum, Xdim, Ydim, Zdim, ...
                XdimInner, YdimInner, ZdimInner, ...
                availableGlutamate, AMPAWeights, AMPAWeights1D, inHz, ...
                preIndexs1D, measurementTrange, measurementTrange, ...
                finalHeadroomStd, Trange(1), sf, HzSf, spikeTrange(1), ...
                HzMax, AMPAMax, perturbationString(perturbation,0,1), ...
                directory, perturbationString(perturbation,1,0));
        if (postprocess_RiskySynapses)
            [~, figNum] = newFigure(figNum, true);
            h = scatterhist(headroomFinal(timeToHeadroomConverged), ...
                timeToHeadroom(timeToHeadroomConverged), ...
                'Kernel', 'on', 'Marker','.');
            set(h(1),'yscale','log');
            ylim([min(timeToHeadroom(timeToHeadroomConverged)) ...
                max(timeToHeadroom(timeToHeadroomConverged))+max(timeToHeadroom(timeToHeadroomConverged))/5]);
            set(h(3),'xscale','log');
            title(['Headroom vs Time to headroom',perturbationString(perturbation,0,1)]);
            xlabel('headroom'); ylabel('time to headroom');
            print([directory,'headroomVsTimeToHeadroom',perturbationString(perturbation,1,0)],'-dpng');
            legend off;
            
            riskyGroup = zeros(1,numel(headroomFinal));
            riskyGroup(headroomFinal<headroomRisk ...
                & timeToHeadroom>=median(timeToHeadroom)) = 1;
            riskyGroup(headroomFinal>=headroomRisk ...
                & timeToHeadroom<median(timeToHeadroom)) = 2; 
            riskyGroup(headroomFinal>=headroomRisk ...
                & timeToHeadroom>=median(timeToHeadroom)) = 3;
            
            [~, figNum] = newFigure(figNum, true);
            hp1 = uipanel('position', [0.0 0.5 0.5 0.5]);
            h = scatterhist(headroomFinal(timeToHeadroomConverged), ...
                timeToHeadroom(timeToHeadroomConverged), ...
                'Group', riskyGroup(timeToHeadroomConverged), 'Kernel', 'on', 'Parent', hp1, ...
                'Marker','.');
            set(h(1),'yscale','log');
            set(h(3),'xscale','log');
            ylim([min(timeToHeadroom(timeToHeadroomConverged)) ...
                max(timeToHeadroom(timeToHeadroomConverged))+max(timeToHeadroom(timeToHeadroomConverged))/5]);
            xlabel('headroom'); ylabel('time to headroom');
%                     legend off;

            hp2 = uipanel('position', [0.5 0.5 0.5 0.5]);
            tempHz = HzFinal(preIndexs1D);
            scatterhist(AMPAWeights1D(timeToHeadroomConverged), ...
                tempHz(timeToHeadroomConverged), ...
                'Group', riskyGroup(timeToHeadroomConverged), 'Kernel', 'on', 'Parent', hp2, ...
                'Marker','.');
            xlim([0 AMPAMax]); xlabel('glutamate bound to AMPA');
            ylim([0 HzMax]); ylabel('in Hz');
            legend off;

            hp3 = uipanel('position', [0.0 0.0 0.5 0.5]);
            tempCleft = sum(cleftGlutamate((measurementTrange(1)/sf)-(Trange(1)/sf): ...
                (measurementTrange(end)/sf)-(Trange(1)/sf),:,:,:));
            scatterhist(AMPAWeights1D(timeToHeadroomConverged), ...
                tempCleft(timeToHeadroomConverged), ...
                'Group', riskyGroup(timeToHeadroomConverged), 'Kernel', 'on', 'Parent', hp3, ...
                'Marker','.');
            xlim([0 AMPAMax]); xlabel('glutamate bound to AMPA');
            ylabel('glutamate flux');
            legend off;

            hp4 = uipanel('position', [0.5 0.0 0.5 0.5]);
            scatterhist(tempHz(timeToHeadroomConverged), ...
                tempCleft(timeToHeadroomConverged), ...
                'Group', riskyGroup(timeToHeadroomConverged), 'Kernel', 'on', 'Parent', hp4, ...
                'Marker','.');
            xlim([0 HzMax]); xlabel('in Hz');
            ylabel('glutamate flux');
            legend off;

            print([directory,'riskySynapses',perturbationString(perturbation,1,0)],'-dpng');
            clear tempHz;
            
            [~, figNum] = newFigure(figNum, true);
            bar([0,1,2,3],[mean(tempCleft(riskyGroup==0 & timeToHeadroomConverged)), ...
                mean(tempCleft(riskyGroup==1 & timeToHeadroomConverged)), ...
                mean(tempCleft(riskyGroup==2 & timeToHeadroomConverged)), ...
                mean(tempCleft(riskyGroup==3 & timeToHeadroomConverged))]);
            title('Glutamate flux for each risky group');
            xlabel('risky group');
            ylabel('mean glutamate flux');
            
            print([directory,'riskySynapsesCleftGlutamate',perturbationString(perturbation,1,0)],'-dpng');
            clear tempCleft;
        end
    end
    %%
    if (postprocess_Perturbation)
        if (~perturbation)
            % not perturbation yet so store previous data needed
            if (postprocess_PerturbationHz)
                beforePerturbationHz = HzFinal;
            end
            if (postprocess_PerturbationAMPA)
                beforePerturbationAMPAWeights1D = AMPAWeights1D;
            end
            beforeHeadroomFinal = headroomFinal;
            beforeTimeToHeadroom = timeToHeadroom;
            beforeTimeToHeadroomConverged = timeToHeadroomConverged;
            beforeRiskyGroup = riskyGroup;
        else
            afterHeadroomFinal = headroomFinal;
            afterTimeToHeadroom = timeToHeadroom;
            afterTimeToHeadroomConverged = timeToHeadroomConverged;
            clear headroomFinal timeToHeadroom timeToHeadroomConverged;
            diffHeadroomFinal = afterHeadroomFinal - ...
                beforeHeadroomFinal;
            diffTimeToHeadroom = afterTimeToHeadroom - ...
                beforeTimeToHeadroom;
            headroomConverged = beforeTimeToHeadroomConverged & ...
                afterTimeToHeadroomConverged;
            
            % now perturbation, so process
            for p=1:3 % 1=Hz, 2=AMPA, 3=both
                if (p==1)
                    if (postprocess_PerturbationHz)
                        afterPerturbationHz = HzFinal;
                        diffPerturbationHz = afterPerturbationHz - ...
                            beforePerturbationHz;
                        perturbationType = 'Hz';
                        clear HzFinal;
    
                        beforePerturbation = beforePerturbationHz(preIndexs1D);
                        afterPerturbation = afterPerturbationHz(preIndexs1D);
                        diffPerturbation = diffPerturbationHz(preIndexs1D);
                    else
                        continue;
                    end
                elseif (p==2)
                    if (postprocess_PerturbationAMPA)
                        afterPerturbationAMPAWeights1D = AMPAWeights1D;
                        diffPerturbationAMPA = afterPerturbationAMPAWeights1D - ...
                            beforePerturbationAMPAWeights1D;
                        perturbationType = 'AMPA';
                        clear AMPAWeights1D;
    
                        beforePerturbation = beforePerturbationAMPAWeights1D;
                        afterPerturbation = afterPerturbationAMPAWeights1D;
                        diffPerturbation = diffPerturbationAMPA;
                    else
                        continue;
                    end
                elseif (p==3)
                    perturbationType = 'Both';
                else
                    continue;
                end
                
                % Hz or AMPA perturbations separately
                if (p < 3)
                    %% Plot perturbation
                    [~, figNum] = newFigure(figNum, true);
                    subplot(2,2,1);
                    bar(diffPerturbation);
                    xlim([0 numel(diffPerturbation)]);
                    title([perturbationType,' perturbation']);
                    xlabel('pre-synaptic afferent'); ylabel('perturbation');
                    subplot(2,2,2);
                    histogram(diffPerturbation);
                    title([perturbationType,' perturbation']);
                    xlabel('perturbation'); ylabel('count');
                    subplot(2,2,3);
                    scatter(beforePerturbation, afterPerturbation,'.');
                    if (p==1)
                        xlim([0 HzMax]); ylim([0 HzMax]);
                    elseif (p==2)
                        xlim([0 AMPAMax]); ylim([0 AMPAMax]);
                    end
                    h=lsline;
                    set(h,'color','r');
                    R=corrcoef(beforePerturbation, afterPerturbation);
                    R_squared=R(2)^2;
                    title([perturbationType,...
                        ' before and after perturbation ', '(R squared= ', ...
                        num2str(R_squared), ')']);
                    xlabel(['before ',perturbationType]);
                    ylabel(['after ',perturbationType]);

                    print([directory,'perturbation',perturbationType],'-dpng');
                    clear h R R_squared;
                    %% Compare before and after
                    [~, figNum] = newFigure(figNum, true);

                    subplot(2,3,1);
                    scatter(beforeHeadroomFinal(headroomConverged), ...
                        afterHeadroomFinal(headroomConverged),'.');
                    h=lsline;
                    xlim([0 AMPAMax]); ylim([0 AMPAMax]);
                    set(h,'color','r');
                    R=corrcoef(beforeHeadroomFinal(headroomConverged), ...
                        afterHeadroomFinal(headroomConverged));
                    R_squared=R(2)^2;
                    title({['Headroom before and after ',perturbationType,...
                        ' perturbation'], ...
                        ['(R squared= ', num2str(R_squared), ')']});
                    xlabel('before'); ylabel('after');

                    subplot(2,3,4);
                    scatter(beforeTimeToHeadroom(headroomConverged), ...
                        afterTimeToHeadroom(headroomConverged),'.');
                    set(gca,'xscale','log');
                    set(gca,'yscale','log');
                    h=lsline;
                    xlim([min(beforeTimeToHeadroom(headroomConverged)) ...
                        max(beforeTimeToHeadroom(headroomConverged))]);
                    ylim([min(afterTimeToHeadroom(headroomConverged)) ...
                        max(afterTimeToHeadroom(headroomConverged))]);
                    set(h,'color','r');
                    R=corrcoef(beforeTimeToHeadroom(headroomConverged), ...
                        afterTimeToHeadroom(headroomConverged));
                    R_squared=R(2)^2;
                    title({['Time to headroom before and after ',perturbationType,...
                        'perturbation'], ...
                        ['(R squared= ', num2str(R_squared), ')']});
                    xlabel('before'); ylabel('after');

                    subplot(2,3,2);
                    scatter(diffPerturbation(headroomConverged), ...
                        afterHeadroomFinal(headroomConverged),'.'); 
                    h=lsline;
                    if (p==1)
                        xlim([-HzMax HzMax]); 
                    elseif(p==2)
                        xlim([-AMPAMax AMPAMax]); 
                    end
                    ylim([0 AMPAMax]);
                    set(h,'color','r');
                    R=corrcoef(diffPerturbation(headroomConverged), ...
                        afterHeadroomFinal(headroomConverged));
                    R_squared=R(2)^2;
                    title({[perturbationType,' perturbation vs headroom'], ...
                        ['(R squared= ', num2str(R_squared), ')']});
                    xlabel([perturbationType,' perturbation']); 
                    ylabel('after');

                    subplot(2,3,5);
                    scatter(diffPerturbation(headroomConverged), ...
                        afterTimeToHeadroom(headroomConverged),'.');
                    set(gca,'yscale','log');
                    h=lsline;
                    if (p==1)
                        xlim([-HzMax HzMax]);
                    elseif (p==2)
                        xlim([-AMPAMax AMPAMax]);
                    end
                    ylim([min(afterTimeToHeadroom(headroomConverged)) ...
                        max(afterTimeToHeadroom(headroomConverged))]);
                    set(h,'color','r');
                    R=corrcoef(diffPerturbation(headroomConverged), ...
                        afterTimeToHeadroom(headroomConverged));
                    R_squared=R(2)^2;
                    title({[perturbationType,' perturbation vs time to headroom'], ...
                        ['(R squared= ', num2str(R_squared), ')']});  
                    xlabel([perturbationType,' perturbation']);
                    ylabel('after');    

                    subplot(2,3,3);
                    scatter(diffPerturbation(headroomConverged), ...
                        diffHeadroomFinal(headroomConverged),'.'); 
                    h=lsline;
                    if (p==1)
                        xlim([-HzMax HzMax]); 
                    elseif (p==2)
                        xlim([-AMPAMax AMPAMax]); 
                    end
                    ylim([-AMPAMax AMPAMax]);
                    set(h,'color','r');
                    R=corrcoef(diffPerturbation(headroomConverged), ...
                        diffHeadroomFinal(headroomConverged));
                    R_squared=R(2)^2;
                    title({[perturbationType,' perturbation vs headroom difference '], ...
                        ['(R squared= ', num2str(R_squared), ')']});
                    xlabel([perturbationType,' perturbation']);
                    ylabel('headroom difference');

                    subplot(2,3,6);
                    scatter(diffPerturbation(headroomConverged), ...
                        diffTimeToHeadroom(headroomConverged),'.');
                    set(gca,'yscale','log');
                    h=lsline;
                    if (p==1)
                        xlim([-HzMax HzMax]);
                    elseif (p==2)
                        xlim([-AMPAMax AMPAMax]);
                    end
                    ylim([min(diffTimeToHeadroom(headroomConverged)) ...
                        max(diffTimeToHeadroom(headroomConverged))]);
                    set(h,'color','r');
                    R=corrcoef(diffPerturbation(headroomConverged), ...
                        diffTimeToHeadroom(headroomConverged));
                    R_squared=R(2)^2;
                    title({[perturbationType,' perturbation vs time to ',...
                        'headroom difference'], ...
                        ['(R squared= ', num2str(R_squared), ')']});  
                    xlabel([perturbationType,' perturbation']);
                    ylabel('time to headroom difference'); 

                    print([directory,'perturbation',perturbationType,'Comparison'],...
                        '-dpng');
                    clear h R R_squared;
                    %% Glutamate flux as a function of perturbation
                    [~, figNum] = newFigure(figNum, true);
                    
                    [glutamateFluxPerturbation, glutamateFluxPerturbationMean, ...
                        glutamateFluxPerturbationMeanBinned, ...
                        glutamateFluxPerturbationP] = ...
                        glutamateFluxVsPerturbation(cleftGlutamate, ...
                        afterTimeToHeadroom, diffPerturbation, ...
                        headroomConverged, p, HzMax, AMPAMax, ...
                        perturbationType, '');

                    print([directory,'perturbation',perturbationType,...
                        'GlutamateFlux'],'-dpng');
                    
                    glutamateFluxPerturbationRisk = cell(1,4);
                    glutamateFluxPerturbationMeanRisk = cell(1,4);
                    glutamateFluxPerturbationMeanBinnedRisk = cell(1,4);
                    glutamateFluxPerturbationPRisk = cell(1,4);
                    for group=0:3
                        [~, figNum] = newFigure(figNum, true);
                    
                        [glutamateFluxPerturbationRisk{group+1}, ...
                            glutamateFluxPerturbationMeanRisk{group+1}, ...
                            glutamateFluxPerturbationMeanBinnedRisk{group+1}, ...
                            glutamateFluxPerturbationPRisk{group+1}] = ...
                            glutamateFluxVsPerturbation(...
                            cleftGlutamate(:,beforeRiskyGroup==group), ...
                            afterTimeToHeadroom(beforeRiskyGroup==group), ...
                            diffPerturbation(beforeRiskyGroup==group), ...
                            headroomConverged(beforeRiskyGroup==group), ...
                            p, HzMax, AMPAMax, perturbationType, ...
                            [' (risk group ', num2str(group-1),')']);

                        print([directory,'perturbation',perturbationType,...
                            'GlutamateFlux_riskGroup',num2str(group)],'-dpng');
                    end
                end
                % Hz and AMPA perturbations together
                if (p==3)
                    % Correlation of Hz and AMPA perturbations
                    [~, figNum] = newFigure(figNum, false);
                    scatter(diffPerturbationHz(preIndexs1D), ...
                        diffPerturbationAMPA, '.');
                    xlim([-HzMax HzMax]); ylim([-AMPAMax AMPAMax]);
                    title('Hz perturbation vs AMPA perturbation');
                    xlabel('Hz perturbation');
                    ylabel('AMPA perturbation');
                    h=lsline;
                    set(h,'color','r');
                    R=corrcoef(diffPerturbationHz(preIndexs1D), ...
                        diffPerturbationAMPA);
                    R_squared=R(2)^2;
                    title([perturbationType,...
                        ' difference ', '(R squared= ', ...
                        num2str(R_squared), ')']);                    
                    print([directory,'perturbation',perturbationType],'-dpng');                    
                    clear h R R_squared;
                    %%                    
                    HzRange = -140:20:140;
                    HzOffset = 10;
                    AMPARange = -1.4:0.2:1.4;
                    AMPAOffset = 0.1;
                    diffHeadroomBothPerturbationBinned = zeros(numel(HzRange),...
                        numel(AMPARange));
                    cleftGlutamateBothMeanPerturbationBinned = ...
                        zeros(numel(HzRange), numel(AMPARange));
                    cleftGlutamateBothMaxPerturbationBinned = ...
                        zeros(numel(HzRange), numel(AMPARange));
                    cleftGlutamateBothMeanPerturbationBinnedRisky = cell(1,4);
                    cleftGlutamateBothMaxPerturbationBinnedRisky = cell(1,4);
                    for i=1:4
                        cleftGlutamateBothMeanPerturbationBinnedRisky{i} = ...
                            zeros(numel(HzRange), numel(AMPARange));
                        cleftGlutamateBothMaxPerturbationBinnedRisky{i} = ...
                            zeros(numel(HzRange), numel(AMPARange));
                    end
                    clear i;
                    
                    for i=1:numel(HzRange)
                        for j=1:numel(AMPARange)
                            tempHz = diffPerturbationHz(preIndexs1D)...
                                    >=HzRange(i)-HzOffset & ...
                                diffPerturbationHz(preIndexs1D)...
                                    <HzRange(i)+HzOffset;
                            tempAMPA = diffPerturbationAMPA...
                                    >=AMPARange(j)-AMPAOffset & ...
                                diffPerturbationAMPA...
                                    <AMPARange(j)+AMPAOffset;
                            tempAll = tempHz & tempAMPA & headroomConverged';
                            diffHeadroomBothPerturbationBinned(i,j) = ...
                                mean(diffHeadroomFinal(tempAll));
                            cleftGlutamateBothMeanPerturbationBinned(i,j) = ...
                                mean(glutamateFluxPerturbation(tempAll));
                            temp = max(glutamateFluxPerturbation(tempAll));
                            if numel((temp) > 0)
                                cleftGlutamateBothMaxPerturbationBinned(i,j) = ...
                                    temp;
                            end
                            for group=0:3
                                tempAllRisky = tempAll & (beforeRiskyGroup==group)';
                                cleftGlutamateBothMeanPerturbationBinnedRisky{group+1}(i,j) = ...
                                    mean(glutamateFluxPerturbation(tempAllRisky));   
                                temp = max(glutamateFluxPerturbation(tempAllRisky));
                                if (numel(temp) > 0)
                                    cleftGlutamateBothMaxPerturbationBinnedRisky{group+1}(i,j) = ...
                                        temp;
                                end
                            end
                            clear group tempAllRisky;
                        end
                    end
                    
                    [~, figNum] = newFigure(figNum, false);
                    
                    subplot(2,2,1);
                    imagesc(diffHeadroomBothPerturbationBinned);
                    colormap(flipud(hot)); colorbar();
                    set(gca,'ydir','normal');
                    title('Headroom difference vs both perturbations');
                    xlabel('AMPA perturbation');
                    ylabel('Hz perturbation');
                    h = colorbar; ylabel(h, 'headroom difference')
                    yticks([1,3,5,7,8,9,11,13,15]);
                    xticks([1,3,5,7,8,9,11,13,15]);
                    xticklabels({'-1.4','-1.0','-0.6','-0.2','0.0',...
                        '0.2','0.6','0.8','1.0','1.4'});
                    yticklabels({'-140','-100','-60','-20','0',...
                        '20','60','100','140'});
                    
                    subplot(2,2,2);
                    imagesc(cleftGlutamateBothMeanPerturbationBinned);
                    colormap(flipud(hot)); colorbar();
                    set(gca,'ydir','normal');
                    title('Glutamate flux vs both perturbations');
                    xlabel('AMPA perturbation');
                    ylabel('Hz perturbation');
                    h = colorbar; ylabel(h, 'glutamate flux')
                    xticks([1,3,5,7,8,9,11,13,15]);
                    xticklabels({'-1.4','-1.0','-0.6','-0.2','0.0',...
                        '0.2','0.6','0.8','1.0','1.4'});   
                    yticks([1,3,5,7,8,9,11,13,15]);
                    yticklabels({'-140','-100','-60','-20','0',...
                        '20','60','100','140'});      
                    
                    subplot(2,2,3);
                    imagesc(cleftGlutamateBothMaxPerturbationBinned);
                    colormap(flipud(hot)); colorbar();
                    set(gca,'ydir','normal');
                    title('Glutamate peak vs both perturbations');
                    xlabel('AMPA perturbation');
                    ylabel('Hz perturbation');
                    h = colorbar; ylabel(h, 'glutamate peak')
                    xticks([1,3,5,7,8,9,11,13,15]);
                    xticklabels({'-1.4','-1.0','-0.6','-0.2','0.0',...
                        '0.2','0.6','0.8','1.0','1.4'});   
                    yticks([1,3,5,7,8,9,11,13,15]);
                    yticklabels({'-140','-100','-60','-20','0',...
                        '20','60','100','140'});               
                    
                    print([directory,'perturbation',perturbationType,'Comparison'],'-dpng');
                    
                    [~, figNum] = newFigure(figNum, true);
                    
                    for group=1:4
                        subplot(2,2,group);
                        imagesc(cleftGlutamateBothMeanPerturbationBinnedRisky{group});
                        colormap(flipud(hot)); colorbar();
                        set(gca,'ydir','normal');
                        title(['Glutamate flux vs both perturbations (risk group ',...
                            num2str(group-1),')']);
                        xlabel('AMPA perturbation');
                        ylabel('Hz perturbation');
                        h = colorbar; ylabel(h, 'glutamate flux')
                        xticks([1,3,5,7,8,9,11,13,15]);
                        xticklabels({'-1.4','-1.0','-0.6','-0.2','0.0',...
                            '0.2','0.6','0.8','1.0','1.4'});   
                        yticks([1,3,5,7,8,9,11,13,15]);
                        yticklabels({'-140','-100','-60','-20','0',...
                            '20','60','100','140'});
                    end
                    
                    print([directory,'perturbation',perturbationType,'ComparisonRiskyMean'],'-dpng');
                    
                    [~, figNum] = newFigure(figNum, true);
                    
                    for group=1:4
                        subplot(2,2,group);
                        imagesc(cleftGlutamateBothMaxPerturbationBinnedRisky{group});
                        colormap(flipud(hot)); colorbar();
                        set(gca,'ydir','normal');
                        title(['Glutamate peak vs both perturbations (risk group ',...
                            num2str(group-1),')']);
                        xlabel('AMPA perturbation');
                        ylabel('Hz perturbation');
                        h = colorbar; ylabel(h, 'glutamate peak')
                        xticks([1,3,5,7,8,9,11,13,15]);
                        xticklabels({'-1.4','-1.0','-0.6','-0.2','0.0',...
                            '0.2','0.6','0.8','1.0','1.4'});   
                        yticks([1,3,5,7,8,9,11,13,15]);
                        yticklabels({'-140','-100','-60','-20','0',...
                            '20','60','100','140'});
                    end
                    
                    print([directory,'perturbation',perturbationType,'ComparisonRiskyMax'],'-dpng');
                    
                    clear HzRange HzOffset AMPARange AMPAOffset i j ...
                        tempHz tempAMPA tempCleft tempAll;
                end
            end
        end
    end
end
%% Functions
function [h, figNum] = newFigure(figNum, maximize)
    figNum = figNum + 1;
    h = figure(figNum);
    clf;
    if (maximize)
        set(h, 'Position', get(0,'Screensize'));
    end
end
function [ret] = perturbationString(perturbation, file, label)
    if (perturbation)
        if (file)
            ret = '_perturbation';
        elseif (label)
            ret = ' (Perturbation) ';
        end
    else
        ret = '';
    end
end
function [ret, Xdim, Ydim, Zdim] = loadSpikes(directory, file, fileExt, ...
    spikesFilter, spikesFilterTmin, spikesFilterTmax, Tmin, Tmax, dt)
    fid = fopen([directory,file,fileExt],'r');
    Xdim = fread(fid, 1, 'int');
    Ydim = fread(fid, 1, 'int');
    Zdim = fread(fid, 1, 'int');
    % temp(1,:) is neuron id and temp(2,:) is spike time
    temp = fread(fid, Inf, 'int'); % have to load all sadly
    if (mod(numel(temp),2) ~= 0)
        temp(end) = [];
    end
    temp = reshape(temp, [2, numel(temp)/2]);
    fclose(fid);
    clear fid;
    if (spikesFilter)
        temp(:,temp(2,:)<spikesFilterTmin/dt) = []; % filter out less than Tmin
        temp(:,temp(2,:)>spikesFilterTmax/dt) = []; % filter out greater than Tmax
    else
        temp(:,temp(2,:)<Tmin/dt) = []; % filter out greater than Tmin
        temp(:,temp(2,:)>Tmax/dt) = []; % filter out greater than Tmax
    end
    % make sure some spikes for every neuron, required for the following
    if (size(unique(temp(1,:)),2) ~= Xdim*Ydim*Zdim)
        error('Some neurons are missing spikes');
    end
    temp(1,:) = temp(1,:)+1; % because of 0 indexing
    temp = sortrows(temp'); % Soft by neuron id
    [~, temp_i] = unique(temp(:,1)); % First spike of each neuron (for those that have spikes)
    spikesN = circshift(temp_i,-1) - temp_i; % Number of spikes for each neuron (for those that have spikes)
    spikesN(end) = size(temp,1) - sum(spikesN(1:end-1)); % fix last neuron
    % {3D cell: X, Y, Z} (1D: spike times)
    ret = cell(Xdim,Ydim,Zdim);
    i=1; % Neuron id counter
    k=1; % Position in temp_i and spikesN
    for z=1:Zdim
        for y=1:Ydim
            for x=1:Xdim
                % If this neuron has spikes
                if (temp(temp_i(k),1) == i)
                    ret{x,y,z} = ...
                        temp(temp_i(k):temp_i(k)+spikesN(k)-1,2);
                    k=k+1;
                end
                i=i+1;
            end
        end
    end
end               
function [ret, XdimInner, YdimInner, ZdimInner] = load4D(directory, file, ...
    fileExt, Tmin, Tmax, sf)
    fid = fopen([directory,file,fileExt],'r');
    XdimInner = fread(fid, 1, 'int');
    YdimInner = fread(fid, 1, 'int');
    ZdimInner = fread(fid, 1, 'int');
    fseek(fid, ((XdimInner*YdimInner*ZdimInner)*(Tmin/sf))*4, 'cof');
    ret = fread(fid, (XdimInner*YdimInner*ZdimInner)*((Tmax-Tmin)/sf), 'float');
    fclose(fid);
    clear fid;
    % 4D: time, X, Y, Z
    ret = reshape(ret(1:floor(numel(ret)/...
        ((XdimInner*YdimInner*ZdimInner)))*XdimInner*YdimInner*ZdimInner), ...
        [XdimInner, YdimInner, ZdimInner, ...
        floor(numel(ret)/(XdimInner*YdimInner*ZdimInner))]);
    ret = permute(ret, [4, 1, 2, 3]);
end
function [ret, ret1D] = load2D(directory, file, fileExt, XdimInner, ...
    YdimInner, ZdimInner)
    fid = fopen([directory,file,fileExt],'r');
    % 4D: [1=pre 2=post], X, Y, Z
    ret = fread(fid, [2, Inf], 'int');
    ret1D = ret(1,:,:,:);
    fclose(fid);
    clear fid;
    ret = reshape(ret, [2, XdimInner, YdimInner, ZdimInner]);
end
function [ret, ret1D] = load3D(directory, file, fileExt, XdimInner, ...
    YdimInner, ZdimInner)
    fid = fopen([directory,file,fileExt],'r');
    % 3D: X, Y, Z
    ret1D = fread(fid, Inf, 'float');
    fclose(fid);
    clear fid;
    ret = reshape(ret1D, [XdimInner, YdimInner, ZdimInner]);
end
function plotModulation(directory, file, fileExt, titleStr, x, ...
    y, saveFile)
    % mGluR modulation function only first
    fid = fopen([directory,file,fileExt],'r');
    mGluRmodulation = fread(fid, Inf, 'float');
    fclose(fid);
    clear fid;
    plot(0:1/1000:2, mGluRmodulation);
    title(titleStr);
    xlabel(x); ylabel(y);
    print([directory,saveFile],'-dpng');
end
function [ret] = calculateHz(input, Tmin, Tmax, HzSf, Xdim, ...
    Ydim, Zdim, dt)
    ret=zeros(numel(Tmin:HzSf:Tmax-HzSf),Xdim,Ydim,Zdim);
    for x=1:Xdim
        for y=1:Ydim
            for z=1:Zdim
                i=1;
                for t=Tmin:HzSf:Tmax-HzSf
                   ret(i,x,y,z) = ...
                        numel(find(...
                            input{x,y,z}>=t/dt & ...
                            input{x,y,z}<(t+1)/dt...
                        ))*(1.0/HzSf);
                    i=i+1;                                
                end
            end
        end
    end
end
function [ret] = hist3D(range, in)
    ret = zeros(numel(range),size(in,1));
    for i=1:size(in,1)
        temp = in(i,:,:,:);
        ret(:,i) = hist(temp(:),range);
    end
end
function [ret] = histHeadroom3D(range, availableGlutamate, weights)
    ret = zeros(numel(range),size(availableGlutamate,1));  
    for i=1:size(availableGlutamate,1) % over time, glutamate and AMPA are same size
        % calculate the diff/headroom for each synapse and hist it
        tempAvailableGlutamate = availableGlutamate(i,:,:,:);
        ret(:,i) = hist(tempAvailableGlutamate(:)-weights, ...
            range);
    end
end
function [headroomFinal, headroomFinalStd, HzFinal, timeToHeadroom, ...
    timeToHeadroomConverged, figNum] = ...
    plotComponentVsHeadroom(figNum, Xdim, Ydim, Zdim, ...
    XdimInner, YdimInner, ZdimInner, ...
    availableGlutamate, AMPAWeights, AMPAWeights1D, inHz, ...
    preIndexs1D, finalHeadroomT, finalHzT, finalHeadroomStd, Tmin, sf, HzSf, ...
    inputSpikeTimesTmin, HzMax, AMPAMax, additionalTitle, ...
    directory, additionalSave)  
    [~, figNum] = newFigure(figNum, true);   
    subplot(2,2,1);
    headroomFinal = zeros(1,XdimInner*YdimInner*ZdimInner);
    headroomFinalStd = zeros(1,XdimInner*YdimInner*ZdimInner);
    i = 1;
    for x=1:XdimInner
        for y=1:YdimInner
            for z=1:ZdimInner
                tempAvailableGlutamate = availableGlutamate(:,x,y,z);
                headroomFinal(i) = mean(tempAvailableGlutamate( ...
                    ((finalHeadroomT(1)-Tmin)/sf)+1:(finalHeadroomT(end)-Tmin)/sf) - ...
                    AMPAWeights(x,y,z));
                headroomFinalStd(i) = std(tempAvailableGlutamate( ...
                    ((finalHeadroomT(1)-Tmin)/sf)+1:(finalHeadroomT(end)-Tmin)/sf) - ...
                    AMPAWeights(x,y,z));
                i = i + 1;
            end
        end
    end
    scatter(AMPAWeights1D, headroomFinal,'.');
    h=lsline;
    set(h,'color','r');
    R=corrcoef(AMPAWeights1D, headroomFinal);
    R_squared=R(2)^2;
    title(['AMPA vs headroom', additionalTitle, ...
        '(R squared= ', num2str(R_squared), ')']);
    xlabel('AMPA'); ylabel('headroom');
    clear i x y z tempAvailableGlutamate tempAMPA h R R_squared;
    
    subplot(2,2,2);
    HzFinal = zeros(Xdim*Ydim*Zdim,1);
    i=1;
    for x=1:Xdim
        for y=1:Ydim
            for z=1:Zdim
                HzFinal(i) = mean(inHz(((finalHzT(1)-inputSpikeTimesTmin)/HzSf)+1: ...
                    (finalHzT(end)-inputSpikeTimesTmin)/HzSf,x,y,z));
                i=i+1;
            end
        end
    end
    scatter(HzFinal(preIndexs1D), headroomFinal,'.');
    xlim([0 HzMax]);
    h=lsline;
    set(h,'color','r');
    R=corrcoef(HzFinal(preIndexs1D), headroomFinal);
    R_squared=R(2)^2;
    title(['Input Hz vs headroom', additionalTitle, ...
        '(R squared= ', num2str(R_squared), ')']);
    xlabel('Hz'); ylabel('headroom');
    clear i x y z tempPreIndexs h R R_squared;
    
    % Time to headroom
    timeToHeadroom = zeros(1,XdimInner*YdimInner*ZdimInner);
    i = 1;
    for x=1:XdimInner
        for y=1:YdimInner
            for z=1:ZdimInner
                tempAvailableGlutamate = availableGlutamate(:,x,y,z);
                tempHeadroom = tempAvailableGlutamate - AMPAWeights(x,y,z);
                temp = find(tempHeadroom<=headroomFinal(i)...
                    +headroomFinalStd(i)*(finalHeadroomStd/2) & ...
                    tempHeadroom>=headroomFinal(i)...
                    -headroomFinalStd(i)*(finalHeadroomStd/2));
                if (numel(temp) == 0)
                    timeToHeadroom(i) = -9;
                else
                    timeToHeadroom(i) = temp(1);
                end
                i = i + 1;
            end
        end
    end
    
    subplot(2,2,3);
    timeToHeadroomConverged = timeToHeadroom>1; % those that actually had to converge
    scatter(AMPAWeights1D(timeToHeadroomConverged), ...
        timeToHeadroom(timeToHeadroomConverged),'.');
    set(gca,'yscale','log');
    ylim([min(timeToHeadroom(timeToHeadroomConverged)) ...
        max(timeToHeadroom(timeToHeadroomConverged))+...
        max(timeToHeadroom(timeToHeadroomConverged))/5]);
    xlim([0 AMPAMax]);
    h=lsline;
    set(h,'color','r');
    R=corrcoef(AMPAWeights1D(timeToHeadroomConverged), ...
        timeToHeadroom(timeToHeadroomConverged));
    R_squared=R(2)^2;
    title(['AMPA vs time to headroom', additionalTitle, '(R squared= ', num2str(R_squared), ')']);
    xlabel('AMPA'); ylabel('time to headroom');
    clear i tempAvailableGlutamate tempHeadroom temp h R R_squared;
    
    subplot(2,2,4);
    scatter(HzFinal(preIndexs1D(timeToHeadroomConverged)), ...
        timeToHeadroom(timeToHeadroomConverged),'.');
    set(gca,'yscale','log');
    ylim([min(timeToHeadroom(timeToHeadroomConverged)) ...
        max(timeToHeadroom(timeToHeadroomConverged))+...
        max(timeToHeadroom(timeToHeadroomConverged))/5]);
    xlim([0 HzMax]);
    h=lsline;
    set(h,'color','r');
    R=corrcoef(HzFinal(preIndexs1D(timeToHeadroomConverged)), ...
        timeToHeadroom(timeToHeadroomConverged));
    R_squared=R(2)^2;
    title(['Input Hz vs time to headroom', additionalTitle, ...
        '(R squared= ', num2str(R_squared), ')']);
    xlabel('Hz'); ylabel('time to headroom');
    print([directory,'componentsVsHeadroomDynamics',additionalSave],'-dpng');
    clear h R R_squared;
end
function [glutamateFluxPerturbation, glutamateFluxPerturbationMean, ...
    glutamateFluxPerturbationMeanBinned, glutamateFluxPerturbationP] = ...
    glutamateFluxVsPerturbation(cleftGlutamate, afterTimeToHeadroom, ...
    diffPerturbation, headroomConverged, p, HzMax, AMPAMax, perturbationType, ...
    titleStr)
    subplot(10,2,1:2:20); hold on;
    glutamateFluxPerturbation = zeros(1,size(cleftGlutamate,2));
    for i=1:size(cleftGlutamate,2)
        glutamateFluxPerturbation(i) = ...
            sum(cleftGlutamate(1:afterTimeToHeadroom(i),i));
    end
    scatter(diffPerturbation(headroomConverged), ...
        glutamateFluxPerturbation(headroomConverged),'.');
    if (p==1)
        Range = -HzMax+5:2:HzMax-5;
        offset = 5;
    elseif (p==2)
        Range = -AMPAMax+0.1:0.05:AMPAMax-0.1;
        offset = 0.1;
    end
    glutamateFluxPerturbationMean = zeros(1,numel(Range));
    tempDiffPerturbation = diffPerturbation;
    i = 1;
    for R=Range
        temp = tempDiffPerturbation >= R-offset & ...
            tempDiffPerturbation < R+offset;
        glutamateFluxPerturbationMean(i) = ...
            mean(glutamateFluxPerturbation(temp));
        i=i+1;
    end
    plot(Range, glutamateFluxPerturbationMean, 'LineWidth', 2);
    if (p==1)
        xlim([-HzMax HzMax]);
    elseif (p==2)
        xlim([-AMPAMax AMPAMax]);
    end
    title([perturbationType, ' perturbation vs glutamate flux', titleStr]);
    xlabel([perturbationType,' perturbation']);
    ylabel('glutamate flux');
    clear Range offset;

    subplot(10,2,2:2:17);
    if (p==1)
        Range = -140:20:140;
        offset = 10;
    elseif (p==2)
        Range = -1.4:0.2:1.4;
        offset = 0.1;
    end
    glutamateFluxPerturbationMeanBinned = zeros(1,numel(Range));
    tempDiffPerturbation = diffPerturbation;
    glutamateFluxPerturbationP = ones(1,numel(Range));
    % Find little change case for stat tests
    littleChange = glutamateFluxPerturbation(tempDiffPerturbation >= -offset & ...
        tempDiffPerturbation < offset);
    for i=1:numel(Range)
        % Find mean and stat to little change case
        temp = tempDiffPerturbation >= Range(i)-offset & ...
            tempDiffPerturbation < Range(i)+offset;
        glutamateFluxPerturbationMeanBinned(i) = ...
            mean(glutamateFluxPerturbation(temp));
        if (numel(glutamateFluxPerturbation(temp)) > 0 && ...
            numel(littleChange > 0))
            [glutamateFluxPerturbationP(i)] = ...
                ranksum(glutamateFluxPerturbation(temp), ...
                littleChange);
        end
    end
    bar(Range, glutamateFluxPerturbationMeanBinned);
    title([perturbationType,' perturbation vs glutamate flux', titleStr]);
    xlabel([perturbationType,' perturbation']);
    if (p==1)
        xlim([-HzMax+10 HzMax-10]);
        xticks([-140:40:-20,0,20:40:140]);
    elseif (p==2)
        xlim([-AMPAMax AMPAMax]);
        xticks(-1.4:0.2:1.4);
    end
    ylabel('glutamate flux');

    subplot(10,2,20);
    plot(Range, glutamateFluxPerturbationP<0.005,'*');
    set(gca,'XTick',[]); set(gca,'YTick',[]);
    if (p==1)
        xlim([-HzMax+10 HzMax-10]);
    elseif (p==2)
        xlim([-AMPAMax AMPAMax]);
    end
    ylim([0.5 1.5]);
end
>>>>>>> origin/team-A
