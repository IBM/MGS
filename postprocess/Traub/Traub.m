clear all; %close all; 
set(0,'defaulttextinterpreter','latex'); rng('shuffle');
%% Load data parameters
dt=0.0001; % time step in s
sf=0.0050; % sample frequency in s
T=10; % Length of simulation time saving data (excluding spikes) in s
Tmin=0; Tmax=T; % default
TminZoom=1; TmaxZoom=2;
loadData = true;
parameterSearch = false;
animatedVisualization = false;
postprocess_CortexInput = true;
postprocess_CortexInputWave = false;
postprocess_CortexInputFile = true;
postprocess_Spikes = true;
postprocess_Voltages = false;
postprocess_Thresholds = false;
postprocess_SpikeVoltages = false;
postprocess_LFPs = true;
postprocess_LFP_FSIsynapses = false;
postprocess_weights = true;
postprocess_GJs = true;
postprocess_PSPs = false;
postprocess_PlotAll = true;
directory='../../graphs/Traub/';
fileExt='.dat';
% for no parameter search just set to all ranges to zero
vX_range=0;
vX_precision='. 0'; % precision after decimal place
vY_range=0;
vY_precision='. 0';
t_range=0;
peakHz = zeros(numel(vX_range),numel(vY_range),numel(t_range));
for vX=vX_range
    for vY=vY_range
        for t=t_range
            %% Setup the directory and check if it is present
            if (parameterSearch)
                directory=[strrep(num2str(vX,['%',vX_precision,'f\n']),'.','_'),'/',...
                    strrep(num2str(vY,['%',vY_precision,'f\n']),'.','_'),'/',...
                    num2str(t),'/']
            else
                directory % just for feedback
            end
            %% Load data
            if (loadData)
                if (postprocess_CortexInput)
                    if (postprocess_CortexInputWave)
                        fid = fopen([directory,'Wave',fileExt],'r');
                    elseif (postprocess_CortexInputFile)
                        fid = fopen([directory,'Ctx',fileExt],'r');
                    end
                    XdimCtx = fread(fid, 1, 'int');
                    YdimCtx = fread(fid, 1, 'int');
                    ZdimCtx = fread(fid, 1, 'int');
                    temp = fread(fid, (XdimCtx*YdimCtx*ZdimCtx)*(T/sf), 'float');
                    fclose(fid);
                    clear fid;
                    % 4D: time, X, Y, Z
                    temp = reshape(temp, [XdimCtx, YdimCtx, ZdimCtx, numel(temp)/(XdimCtx*YdimCtx*ZdimCtx)]);
                    cortex = permute(temp, [4, 1, 2, 3]);
                    clear temp;
                end
                if (postprocess_Spikes)
                    fid = fopen([directory,'Spike',fileExt],'r');
                    XdimStrSpikes = fread(fid, 1, 'int');
                    YdimStrSpikes = fread(fid, 1, 'int');
                    ZdimStrSpikes = fread(fid, 1, 'int'); % have to load all sadly
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
                    spike = ones(maxSpikes,XdimStrSpikes,YdimStrSpikes,ZdimStrSpikes)*-1;
                    i=1; % Neuron id counter
                    j=1; % Position in temp3
                    k=1; % Position in temp4_i and spikesN
                    for z=1:ZdimStrSpikes
                        for y=1:YdimStrSpikes
                            for x=1:XdimStrSpikes
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
                if (postprocess_Voltages)
                    fid = fopen([directory,'Voltage',fileExt],'r');
                    XdimStr = fread(fid, 1, 'int');
                    YdimStr = fread(fid, 1, 'int');
                    ZdimStr = fread(fid, 1, 'int');
                    temp = fread(fid, (XdimStr*YdimStr*ZdimStr)*(T/sf), 'float');
                    fclose(fid);
                    clear fid;
                    % 4D: time, X, Y, Z
                    temp = reshape(temp, [XdimStr, YdimStr, ZdimStr, numel(temp)/(XdimStr*YdimStr*ZdimStr)]);
                    voltage = permute(temp, [4, 1, 2, 3]);
                    clear temp;
                end
                if (postprocess_Thresholds)
                    fid = fopen([directory,'Threshold',fileExt],'r');
                    XdimStr = fread(fid, 1, 'int');
                    YdimStr = fread(fid, 1, 'int');
                    ZdimStr = fread(fid, 1, 'int');
                    temp = fread(fid, (XdimStr*YdimStr*ZdimStr)*(T/sf), 'float');
                    fclose(fid);
                    clear fid;
                    % 4D: time, X, Y, Z
                    temp = reshape(temp, [XdimStr, YdimStr, ZdimStr, numel(temp)/(XdimStr*YdimStr*ZdimStr)]);
                    threshold = permute(temp, [4, 1, 2, 3]);
                    clear temp;
                end
                if (postprocess_SpikeVoltages)
                    fid = fopen([directory,'SpikeVoltage',fileExt],'r');
                    XdimStr = fread(fid, 1, 'int');
                    YdimStr = fread(fid, 1, 'int');
                    ZdimStr = fread(fid, 1, 'int');
                    temp = fread(fid, (XdimStr*YdimStr*ZdimStr)*(T/sf), 'float');
                    fclose(fid);
                    clear fid;
                    % 4D: time, X, Y, Z
                    temp = reshape(temp, [XdimStr, YdimStr, ZdimStr, numel(temp)/(XdimStr*YdimStr*ZdimStr)]);
                    spikeVoltage = permute(temp, [4, 1, 2, 3]);
                    clear temp;
                end
                if (postprocess_LFPs)
                    fid = fopen([directory,'LFP',fileExt],'r');
                    XdimLFP = fread(fid, 1, 'int');
                    YdimLFP = fread(fid, 1, 'int');
                    ZdimLFP = fread(fid, 1, 'int');
                    temp = fread(fid, (XdimLFP*YdimLFP*ZdimLFP)*(T/sf), 'float');
                    fclose(fid);
                    clear fid;
                    % 4D: time, X, Y, Z
                    temp = reshape(temp, [XdimLFP, YdimLFP, ZdimLFP, numel(temp)/(XdimLFP*YdimLFP*ZdimLFP)]);
                    LFPs = permute(temp, [4, 1, 2, 3]);
                    clear temp;
                end
                if (postprocess_LFP_FSIsynapses)
                    fid = fopen([directory,'LFP_FSIsynapses',fileExt],'r');
                    XdimStr = fread(fid, 1, 'int');
                    YdimStr = fread(fid, 1, 'int');
                    ZdimStr = fread(fid, 1, 'int');
                    temp = fread(fid, (XdimStr*YdimStr*ZdimStr)*(T/sf), 'float');
                    fclose(fid);
                    clear fid;
                    % 4D: time, X, Y, Z
                    temp = reshape(temp, [XdimStr, YdimStr, ZdimStr, numel(temp)/(XdimStr*YdimStr*ZdimStr)]);
                    LFPs_FSIsynapses = permute(temp, [4, 1, 2, 3]);
                    clear temp;
                end
                if (postprocess_weights)
                    warning('Code needs updating to be efficient');
                    if (~postprocess_Spikes && ~postprocess_Voltages && ~postprocess_Thresholds ...
                        && ~postprocess_SpikeVoltages && ~postprocess_LFP_FSIsynapses)
                        warning(['To process weights at least one other set of data should also ', ...
                            'be processed.']);
                        return;
                    end
                    fid = fopen([directory,'Weights',fileExt],'r');
                    numSynapses = zeros(1,XdimStrSpikes*YdimStrSpikes*ZdimStrSpikes);
                    weights = cell(1,XdimStrSpikes*YdimStrSpikes*ZdimStrSpikes);
                    for i=1:XdimStrSpikes*YdimStrSpikes*ZdimStrSpikes
                        numSynapses(i) = fread(fid, 1, 'int');
                        weights{i} = fread(fid, numSynapses(i), 'float')';
                    end
                    fclose(fid);
                    clear fid;
                end
                if (postprocess_GJs)
                    warning('Code needs updating to be efficient');
                    if (~postprocess_Spikes && ~postprocess_Voltages && ~postprocess_Thresholds ...
                        && ~postprocess_SpikeVoltages && ~postprocess_LFP_FSIsynapses)
                        warning(['To process GJs at least one other set of data should also ', ...
                            'be processed.']);
                        return;
                    end
                    fid = fopen([directory,'GJs',fileExt],'r');
                    numGJs = zeros(1,XdimStrSpikes*YdimStrSpikes*ZdimStrSpikes);
                    GJs = cell(1,XdimStrSpikes*YdimStrSpikes*ZdimStrSpikes);
                    for i=1:XdimStrSpikes*YdimStrSpikes*ZdimStrSpikes
                        numGJs(i) = fread(fid, 1, 'int');
                        GJs{i} = fread(fid, numGJs(i), 'float')';
                    end
                    fclose(fid);
                    clear fid;
                end
                % one cell per neurons, where each cell contains 1 row per dt, where
                % columns are synapses
                if (postprocess_PSPs)
                    warning('Code needs updating to be efficient');
                    if (~postprocess_Spikes && ~postprocess_Voltages && ~postprocess_Thresholds ...
                        && ~postprocess_SpikeVoltages && ~postprocess_LFP_FSIsynapses)
                        warning(['To process PSPs at least one other set of data should also ', ...
                            'be processed (in addition to weights).']);
                        return;
                    end
                    if (~postprocess_weights)
                        warning('To process PSPs weights should be processed.');
                        return;
                    end
                    fid = fopen([directory,'PSPs',fileExt],'r');
                    PSPs = cell(1,XdimStr*YdimStr);
                    ti=1;
                    while(~feof(fid))
                        for i=1:XdimStr*YdimStr
                            temp = fread(fid, numSynapses(i), 'float');
                            if (numel(temp) > 0)
                                PSPs{i}(ti,:) = temp;
                            end
                        end
                        ti=ti+1;
                    end
                    fclose(fid);
                    clear temp;
                end
            end
            %% Plot parameters
            Nstr=5; % Number of plots for each dimension in the striatum data
            pmtmNW=4; % The time-halfbandwidth for the PMTM
            pmtmFrange=0.5:0.5:100;
            if (postprocess_CortexInput)
                XminCtx=1; XmaxCtx=XdimCtx;
                YminCtx=1; YmaxCtx=YdimCtx;
                ZminCtx=1; ZmaxCtx=ZdimCtx;
            else
                XminCtx=1; XmaxCtx=1;
                YminCtx=1; YmaxCtx=1;
                ZminCtx=1; ZmaxCtx=1;
            end
            if (postprocess_Spikes)
                XminStrSpikes=1; XmaxStrSpikes=XdimStrSpikes;
                YminStrSpikes=1; YmaxStrSpikes=YdimStrSpikes;
                ZminStrSpikes=1; ZmaxStrSpikes=ZdimStrSpikes;
            else
                XminStr=1; XmaxStr=1;
                YminStr=1; YmaxStr=1;
                ZminStr=1; ZmaxStr=1;
            end
            %% Plot
            if (postprocess_CortexInput)
                figure(1); clf; % Cortex input
                if (postprocess_PlotAll)
                    i=1;
                    for x=1:XdimCtx
                        for y=1:YdimCtx
                            for z=1:ZdimCtx
                                subplot(XdimCtx*YdimCtx, ZdimCtx, i);
                                plot(cortex(:,x,y,z)); 
%                                 title('Cortex input');
                                xlim([(Tmin*(1/dt))*(dt/sf) (Tmax*(1/dt))*(dt/sf)]);
                                i=i+1;
                            end
                        end
                    end
                else
                    plot(cortex(:,1,1,1)); 
                    title('Cortex input');
                    xlim([(Tmin*(1/dt))*(dt/sf) (Tmax*(1/dt))*(dt/sf)]);
                end
                print([directory,'cortex'],'-dpng');
                print([directory,'cortex'],'-depsc');
            end
            if (postprocess_Spikes)
                figure(2); clf; % Spikes
%                 if (postprocess_PlotAll)
%                     segmentDim=XdimStrSpikes;
%                 else
                    segmentDim=XdimStrSpikes/XdimCtx;
%                 end
                Nspikes=segmentDim^3; % Number of neurons in one segment
                maxSpikes=round(maxSpikes/1); % can be used to shrink if too large
                temp = zeros(maxSpikes*Nspikes,2);
                i=1;
                for x=1:segmentDim
                    for y=1:segmentDim
                        for z=1:segmentDim
                            temp((i-1)*maxSpikes+1:i*maxSpikes,1) ...
                                = ones(maxSpikes,1)*i;
                            temp((i-1)*maxSpikes+1:i*maxSpikes,2) ...
                                = spike(1:maxSpikes,x,y,z);
                            i=i+1;
                        end
                    end
                end
                scatter(temp(:,2),temp(:,1));
                xlim([Tmin*(1/dt) Tmax*(1/dt)]);
                ylim([0 Nspikes]);
%                 if (postprocess_PlotAll)
%                     title('Spiking activity (all "segments" of striatum)');
%                 else
                    title('Spiking activity (one "segment" of striatum)');
%                 end
                temp2 = temp(1:maxSpikes*100,:);
                save([directory,'Spike.txt'],'-ascii','temp2');
                print([directory,'spikes'],'-dpng');
                print([directory,'spikes'],'-depsc');
                xlim([TminZoom*(1/dt) TmaxZoom*(1/dt)]);
                print([directory,'spikes_zoom'],'-dpng');
                print([directory,'spikes_zoom'],'-depsc');
                clear temp temp2 i segmentDim Nspikes;
            end
%             Drange = [XmaxStr-XminStr,YmaxStr-YminStr,ZmaxStr-ZminStr];
%             points = rand(Nstr*3,3);
%             points = bsxfun(@times,Drange,points);
%             Dmin = [XminStr,YminStr,ZminStr];
%             points = round(bsxfun(@plus,Dmin,points));
            if (postprocess_Voltages)
                figure(3); clf; % Sub-threshold voltage
                for i=1:Nstr*3
                    subplot(3, Nstr, i);
                    plot(voltage(:,points(i,1),points(i,2),points(i,3)));
                    title('Sub-threshold voltage');
                    xlim([(Tmin*(1/dt))*(dt/sf) (Tmax*(1/dt))*(dt/sf)]);
                end
                print([directory,'sub-voltage'],'-dpng');
                print([directory,'sub-voltage'],'-depsc');
            end
            if (postprocess_Thresholds)
                figure(4); clf; % Threshold
                for i=1:Nstr*3
                    subplot(3, Nstr, i);
                    plot(threshold(:,points(i,1),points(i,2),points(i,3))); 
                    title('Threshold');
                    xlim([(Tmin*(1/dt))*(dt/sf) (Tmax*(1/dt))*(dt/sf)]);
                end
                print([directory,'threshold'],'-dpng');
                print([directory,'threshold'],'-depsc');
            end
            if (postprocess_SpikeVoltages)
                figure(5); clf; % Voltage (with spikes)
                for i=1:Nstr*3
                    subplot(3, Nstr, i);
                    plot(spikeVoltage(:,points(i,1),points(i,2),points(i,3))); 
                    title('Voltage');
                    xlim([(Tmin*(1/dt))*(dt/sf) (Tmax*(1/dt))*(dt/sf)]);
                end
                print([directory,'voltage'],'-dpng');
                print([directory,'voltage'],'-depsc');
            end
            if (postprocess_LFP_FSIsynapses)
                figure(6); clf; % LFPs - inhibitory - individual neurons
                for i=1:Nstr*3
                    subplot(3, Nstr, i);
                    plot(LFPs_FSIsynapses(:,points(i,1),points(i,2),points(i,3))); 
                    title('LFP - synapses');
                    xlim([(Tmin*(1/dt))*(dt/sf) (Tmax*(1/dt))*(dt/sf)]);
                end
                print([directory,'LFP_synapses_ind'],'-dpng');
                print([directory,'LFP_synapses_ind'],'-depsc');
                figure(7); clf; % LFPs - inhibitory
                Tlfp=sum(LFPs_FSIsynapses(:,XminStr:XmaxStr,YminStr:YmaxStr,ZminStr:ZmaxStr),4);
                Tlfp=sum(Tlfp,3);
                Tlfp=sum(Tlfp,2);
                plot(Tlfp);
                title('LFP - synapses');
                print([directory,'LFP_synapses'],'-dpng');
                print([directory,'LFP_synapses'],'-depsc');
                figure(8); clf; % Frequency analysis
                [pxx,f] = pmtm(Tlfp,pmtmNW,pmtmFrange,1/sf,'unity');
                temp = [f,10*log10(pxx)];
                save([directory,'PMTM.txt'],'-ascii','temp');
                pmtm(Tlfp,pmtmNW,pmtmFrange,1/sf,'unity');
                print([directory,'pmtm'],'-dpng');
                print([directory,'pmtm'],'-depsc');
                figure(9); clf;
                [pks, locs] = findpeaks(log10(pxx),f,'MinPeakProminence',0.5);
                peakHz(vX==vX_range, vY==vY_range, t==t_range) = ...
                    locs(pks==max(pks));
                [pxx,f,pxxc] = pmtm(Tlfp,pmtmNW,pmtmFrange,1/sf, ...
                    'unity','ConfidenceLevel',0.95);
                clf;
                plot(f,10*log10(pxx))
                hold on
                plot(f,10*log10(pxxc),'r-.')
                xlim([0 100])
                xlabel('Hz')
                ylabel('dB')
                title('Multitaper PSD Estimate with 95%-Confidence Bounds')
                print([directory,'pmtm_95'],'-dpng');
                print([directory,'pmtm_95'],'-depsc');
                clear Tlfp pxx f temp;
            end
            if (postprocess_LFPs)
                %%
                figure(10); clf; % LFP electrodes
                if (postprocess_PlotAll)
                    i=1;
                    for x=1:XdimLFP
                        for y=1:YdimLFP
                            for z=1:ZdimLFP
                                subplot(XdimLFP*YdimLFP, ZdimLFP, i);
                                plot(LFPs(:,x,y,z)); 
                                xlim([(Tmin*(1/dt))*(dt/sf) (Tmax*(1/dt))*(dt/sf)]);
                                i=i+1;
                            end
                        end
                    end
                else
                    plot(LFPs(:,1,1,1)); 
                    xlim([(Tmin*(1/dt))*(dt/sf) (Tmax*(1/dt))*(dt/sf)]);
                end                    
                print([directory,'LFPs'],'-dpng');
                print([directory,'LFPs'],'-depsc');
                inc = size(LFPs,1);
                if (postprocess_PlotAll)
                    i=1;
                    pxx = zeros(XdimLFP*YdimLFP*ZdimLFP,size(pmtmFrange,2));
                    for x=1:XdimLFP
                        for y=1:YdimLFP
                            for z=1:ZdimLFP
                                Tlfp = LFPs(:,x,y,z);
                                [pxxTemp,f,pxxcTemp] = pmtm(Tlfp,pmtmNW,pmtmFrange,1/sf, ...
                                    'unity','ConfidenceLevel',0.95);
                                pxx(i,:) = pxxTemp / sum(pxxTemp);
                                i=i+1;
                            end
                        end
                    end
                    pxxStd = std(pxx);
                    pxxMean = mean(pxx);
                    figure(11); clf; % Frequency analysis
                    X=[pmtmFrange,fliplr(pmtmFrange)];
                    Y=[10*log10(pxxMean+pxxStd),fliplr(10*log10(pxxMean-pxxStd))];
                    fill(X,Y,'r','LineStyle','none');
                    hold on;
                    plot(0.5:0.5:100,10*log10(pxxMean),'r','LineWidth',1.5);
                    alpha(0.35);
                    xlabel('Frequency (Hz)');
                    ylabel('Channel-wise normalized Power Spectral Density');
                    print([directory,'pmtm_std'],'-dpng');
                    print([directory,'pmtm_std'],'-depsc');
                else
                    Tlfp = LFPs(:,1,1,1);
                    [pxx,f,pxxc] = pmtm(Tlfp,pmtmNW,pmtmFrange,1/sf, ...
                        'unity','ConfidenceLevel',0.95);
                    figure(11); clf; % Frequency analysis
                    plot(0.5:0.5:100,10*log10(pxx))
                    xlim([0 100])
                    xlabel('Hz')
                    ylabel('Power/frequency (dB/Hz)')
                    title('Multitaper PSD Estimate')
                    print([directory,'pmtm'],'-dpng');
                    print([directory,'pmtm'],'-depsc');
                    plot(f,10*log10(pxx))
                    hold on
                    plot(f,10*log10(pxxc),'r-.')
                    xlim([0 100])
                    xlabel('Hz')
                    ylabel('Averaged and normalized (dB/Hz)')
                    title('Multitaper PSD Estimate with 95%-Confidence Bounds')
                    print([directory,'pmtm_95'],'-dpng');
                    print([directory,'pmtm_95'],'-depsc');
                end
                %%
                clear inc Tlfp pxx f pxxc temp;
            end
            if (postprocess_weights)
                warning('Code needs updating to be efficient');
                figure(13); clf;
                temp = [];
                for i=1:YdimStrSpikes*XdimStrSpikes*ZdimStrSpikes
                    temp = [temp, weights{i}];
                end
                histogram(temp(:));
                title('Synaptic weights');
                print([directory,'weights'],'-dpng');
                print([directory,'weights'],'-depsc');
                clear temp;
            end
            if (postprocess_GJs)
                warning('Code needs updating to be efficient');
                figure(14); clf;
                temp = [];
                for i=1:YdimStrSpikes*XdimStrSpikes*ZdimStrSpikes
                    temp = [temp, GJs{i}];
                end
                histogram(temp(:));
                title('Gap junction conductances');
                print([directory,'GJs'],'-dpng');
                print([directory,'GJs'],'-depsc');
                clear temp;
            end
            if (postprocess_PSPs)
                warning('Code needs updating to be efficient');
                figure(15); clf;
                i=1;
                for x=rowMin:rowMax
                    for y=1:colMin:colMax
                        x = randperm(numSynapses(i), 5);
                        subplot(rowMax-rowMin+1, colMax-colMin+1, i); 
                        plot(bsxfun(@times, PSPs{i}(:,x), ...
                            weights{i}(x)));
                        title('PSPs');
                        i=i+1;
                    end
                end
                print([directory,'PSPs'],'-dpng');
                print([directory,'PSPs'],'-depsc');
            end
            %% Animate visualization
            if (animatedVisualization)
                %% LFP
                if (postprocess_LFPs)
                    %% Adjust and normalize the data ready for plotting
                    LFPs_animate = mat2gray(LFPs);
                    LFPs_animate = 1-LFPs_animate;
                    %% Animated LFP time series
                    fig = figure(16); clf;
                    fig.Color = 'black';
                    fig.Position = [50 50 1280 720];%1920 1080];
                    vid = VideoWriter([directory,'LFP_ts.avi']);
                    frameRate = 60;
                    vid.FrameRate = frameRate;
                    vid.Quality = 100;
                    open(vid);
                    Nstr = 20;
                    XmaxLFP=XdimLFP;
                    XminLFP=0;
                    n = randi(XmaxLFP-XminLFP,1,Nstr) + XminLFP;
                    temp = zeros(10,size(LFPs_animate,1));
                    % Get data
                    for i=1:Nstr
                       temp(i,:) = LFPs_animate(:,n(i),n(i),n(i));
                    end
                    % Normalize
                    temp = temp / max(temp(:));
                    % Offset
                    for i=1:Nstr
                        temp(i,:) = temp(i,:) + i - 1;
                    end
                    % Animate
                    for ti=linspace(Tmin+2,size(temp,2),frameRate*Tmax)
                        plot(temp(:,round(1:ti))');
                        axis([0 size(temp,2) 0 Nstr]);
                        ax = gca; ax.XTick = []; ax.YTick = []; ax.ZTick = []; axis off;
                        title(['EEG (time=',num2str(ti/1000,'%01.3f'),'s)'],'FontSize',20);
                        frame = getframe(fig);
                        writeVideo(vid,frame);
                    end        
                    close(vid);
                    clear fig vid frameRate N n temp frame;
                    %% Animated 3D LFP
                    XmaxLFP=XdimLFP;
                    XminLFP=1;
                    YmaxLFP=YdimLFP;
                    YminLFP=1;
                    ZmaxLFP=ZdimLFP;
                    ZminLFP=1;
                    fig = figure(17); clf;
                    fig.Color = 'black';
                    fig.Position = [50 50 1280 720];%1920 1080];
                    vid = VideoWriter([directory,'LFP_3D.avi']);
                    vid.FrameRate = 10;
                    vid.Quality = 100;
                    open(vid);
                    sli = 0:1:5;%1:1:27;
                    i = linspace(0,4,5);%1,27,5);%[1,25,50,75,27];
                    [X, Y] = meshgrid(i,i);                         
                    x = [X(:) X(:)]';                                
                    y = [Y(:) Y(:)]';
                    z = [repmat(i(1),1,length(x)); repmat(i(end),1,length(x))];
                    col = ':w';
                    r=0;
                    for ti=Tmin+1:Tmax/sf
                        clf; hold on;
                        % Slices
                        h=slice(permute(LFPs_animate(ti,XminLFP:XmaxLFP,YminLFP:YmaxLFP,ZminLFP:ZmaxLFP),[2,3,4,1]),sli,sli,sli);
                        for i=1:numel(sli)*3
                            cdata = get(h(i),'cdata');
                            set(h(i),'EdgeColor','none',...
                                'AlphaData',(cdata-0.1)*(cdata>0.1),...
                                'AlphaDataMappin','none','FaceAlpha','flat');
                        end
                        ax = gca; ax.XTick = []; ax.YTick = []; ax.ZTick = []; axis off;
                        title(['LFP (time=',num2str(ti/sf,'%01.3f'),'s)'],'FontSize',20);
                        % Grid
                        view(3);
                        plot3(x,y,z,col); plot3(y,z,x,col); plot3(z,x,y,col);
                        % Rotate, zoom, and color
                        camzoom(0.8);
                        camorbit(r,0,'axis');
                        r=r+1;
                        if (r==360)
                            r=0;
                        end
                        % Add the figure as a frame in the movie
                        frame = getframe(fig);
                        writeVideo(vid,frame);
                    end
                    close(vid);
                    clear fig vid sli i X Y x y z col r t h ax frame;
                end
            end
        end
    end
end