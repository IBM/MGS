clear variables; %close all;
set(0,'defaulttextinterpreter','latex'); rng('shuffle');
fontsize=6;
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
fileExt='.dat';
% time frames
T=0:100; % Length of simulation time saving data (excluding spikes) in s to load and process
Tperturbation=100:200; % same as above but for a perturbation
postprocess_InputSpikesFilter=true; % whether to filter input spikes
inputSpikeTimesT=49:T(end); % time window to filter
inputSpikeTimesTperturbation=149:Tperturbation(end); % same as above but for a perturbation
postprocess_OutputSpikesFilter=true; % whether to filter output spikes
outputSpikeTimesT=inputSpikeTimesT; % time window to filter
outputSpikeTimesTperturbation=inputSpikeTimesTperturbation; % same as above but for a perturbation
finalMeasurementT=inputSpikeTimesT; % time window to calculate final measurements
finalMeasurementTperturbation=inputSpikeTimesTperturbation; % same as above but for a perturbation
% additional parameters
finalHeadroomStd=1; % accuracy of final headroom for determining time to headroom
headroomRisk=0.4; % the headroom demarcation line for risky synapses
%% Whether to load/process different data sources
postprocess_Glutamate = true;
glutamate_prep = 'Glutamate_';
postprocess_PreIndexs = true;
postprocess_InputSpikes = true;
postprocess_Neurotransmitter = true;%false;
postprocess_AvailableNeurotransmitter = true;
postprocess_CB1R = true;
postprocess_CB1Runbound = true;
postprocess_CB1Rcurrent = true;%false;
postprocess_GoodwinX = false;
postprocess_GoodwinY = false;
postprocess_GoodwinZ = false;
postprocess_CleftAstrocyteNeurotransmitter = true;
postprocess_CleftAstrocyteeCB = true;%false;
postprocess_AMPAWeights = true;
postprocess_AMPA = true;%false;
postprocess_mGluR5 = true;%false;
postprocess_NMDARopen = true;%false;
postprocess_NMDARCacurrent = true;%false;
postprocess_Ca = true;%false;
postprocess_eCB = true;%false;
postprocess_OutputSpikes = true;%false;

postprocess_Headroom = true;
postprocess_RiskySynapses = false;%true;

postprocess_Perturbation = false;%true;
postprocess_PerturbationHz = false;%true;
postprocess_PerturbationAMPA = false;%true;
%%
for perturbation=0:(1*postprocess_Perturbation)
    clear glutamateInputSpike glutamateNeurotransmitter glutamateAvailableNeurotransmitter ...
        glutamatePreIndexs glutamatePreIndexs1D glutamateCB1R glutamateCB1Runbound ...
        glutamateCB1Rcurrent glutamateCleftGlutamate glutamateAMPA glutamateAMPAWeights ...
        glutamateAMPAWeights1D glutamatemGluR5 glutamateCa glutamateeCB ...
        GABAInputSpike GABANeurotransmitter GABAAvailableNeurotransmitter ...
        GABAPreIndexs GABAPreIndexs1D GABACB1R GABACB1Runbound ...
        GABACB1Rcurrent GABACleftGlutamate GABAAMPA GABAAMPAWeights ...
        GABAAMPAWeights1D GABAmGluR5 GABACa GABAeCB ...
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
        if (postprocess_Glutamate)
            [glutamateInputSpike, Xdim, Ydim, Zdim] = loadSpikes(directory, ...
                [glutamate_prep, 'PoissonSpikes'], fileExt, postprocess_InputSpikesFilter, ...
                spikeTrange(1), spikeTrange(end), Trange(1), Trange(end), dt);
        end
    end
    if (postprocess_Neurotransmitter) % do before PreIndexs to get XdimInner etc.
        if (postprocess_Glutamate)
            [glutamateNeurotransmitter, XdimInner, YdimInner, ZdimInner] = ...
                load4D(directory, [glutamate_prep, 'BoutonNeurotransmitter'], ...
                fileExt, Trange(1), Trange(end), sf);
        end
    end 
    if (postprocess_PreIndexs) % return to correct order
        if (postprocess_Glutamate)
            [glutamatePreIndexs, glutamatePreIndexs1D] = load2D(directory, ...
                [glutamate_prep, 'BoutonIndexs'], fileExt, XdimInner, YdimInner, ZdimInner);
        end
    end
    if (postprocess_AvailableNeurotransmitter)
        if (postprocess_Glutamate)
            [glutamateAvailableNeurotransmitter, XdimInner, YdimInner, ZdimInner] = ...
                load4D(directory, [glutamate_prep, 'BoutonAvailableNeurotransmitter'], fileExt, Trange(1), ...
                Trange(end), sf);
        end
    end 
    if (postprocess_CB1R)
        if (postprocess_Glutamate)
            [glutamateCB1R, XdimInner, YdimInner, ZdimInner] = ...
                load4D(directory, [glutamate_prep, 'BoutonCB1R'], fileExt, Trange(1), ...
                Trange(end), sf);
        end 
    end
    if (postprocess_CB1Runbound)
        if (postprocess_Glutamate)
            [glutamateCB1Runbound, XdimInner, YdimInner, ZdimInner] = ...
                load4D(directory, [glutamate_prep, 'BoutonCB1Runbound'], fileExt, Trange(1), ...
                Trange(end), sf);
        end
    end 
    if (postprocess_CB1Rcurrent)
        if (postprocess_Glutamate)
            [glutamateCB1Rcurrent, XdimInner, YdimInner, ZdimInner] = ...
                load4D(directory, [glutamate_prep, 'BoutonCB1Rcurrent'], fileExt, Trange(1), ...
                Trange(end), sf);
        end
    end 
    if (postprocess_GoodwinX)
        [GoodwinX, XdimInner, YdimInner, ZdimInner] = ...
            load4D(directory, 'Goodwin_X', fileExt, Trange(1), ...
            Trange(end), sf);
    end
    if (postprocess_GoodwinY)
        [GoodwinY, XdimInner, YdimInner, ZdimInner] = ...
            load4D(directory, 'Goodwin_Y', fileExt, Trange(1), ...
            Trange(end), sf);
    end
    if (postprocess_GoodwinZ)
        [GoodwinZ, XdimInner, YdimInner, ZdimInner] = ...
            load4D(directory, 'Goodwin_Z', fileExt, Trange(1), ...
            Trange(end), sf);
    end    
    if (postprocess_CleftAstrocyteNeurotransmitter)
        if (postprocess_Glutamate)
            [glutamateCleftAstrocyteNeurotransmitter, XdimInner, YdimInner, ZdimInner] = ...
                load4D(directory, [glutamate_prep, 'CleftAstrocyteNeurotransmitter'], fileExt, Trange(1), ...
                Trange(end), sf);
        end
    end 
    if (postprocess_CleftAstrocyteeCB)
        if (postprocess_Glutamate)
            [glutamateCleftAstrocyteeCB, XdimInner, YdimInner, ZdimInner] = ...
                load4D(directory, [glutamate_prep, 'CleftAstrocyteeCB'], fileExt, Trange(1), ...
                Trange(end), sf);
        end
    end  
    if (postprocess_AMPAWeights && postprocess_Glutamate)
        if (perturbation && postprocess_PerturbationAMPA)
            [spineAMPAWeights, spineAMPAWeights1D] = load3D(directory, ...
                [glutamate_prep,'SpineAMPAWeights_',num2str(Tperturbation(1)/dt)], fileExt, ...
                XdimInner, YdimInner, ZdimInner);
        else
            [spineAMPAWeights, spineAMPAWeights1D] = load3D(directory, ...
                [glutamate_prep,'SpineAMPAWeights_1'], fileExt, XdimInner, YdimInner, ...
                ZdimInner);
        end
    end   
    if (postprocess_AMPA && postprocess_Glutamate)
        [spineAMPA, XdimInner, YdimInner, ZdimInner] = ...
            load4D(directory, [glutamate_prep,'SpineAMPA'], fileExt, Trange(1), ...
            Trange(end), sf);
    end
    if (postprocess_mGluR5 && postprocess_Glutamate)
        if (~perturbation)
            [~, figNum] = newFigure(figNum, false);
            plotModulation(directory, [glutamate_prep,'SpinemGluR5modulation'], fileExt, ...
                'Spine mGluR5 Modulation Function', 'mGluR5', 'Ca2+', ...
                [glutamate_prep,'mGluR5modulation'], fontsize);
        end
        [spinemGluR5, XdimInner, YdimInner, ZdimInner] = ...
            load4D(directory, [glutamate_prep,'SpinemGluR5'], fileExt, Trange(1), ...
            Trange(end), sf);
    end
    if (postprocess_NMDARopen && postprocess_Glutamate)
        [spinemNMDARopen, XdimInner, YdimInner, ZdimInner] = ...
            load4D(directory, [glutamate_prep,'SpineNMDARopen'], fileExt, Trange(1), ...
            Trange(end), sf);
    end
    if (postprocess_NMDARCacurrent && postprocess_Glutamate)
        [spinemNMDARCacurrent, XdimInner, YdimInner, ZdimInner] = ...
            load4D(directory, [glutamate_prep,'SpineNMDARCacurrent'], fileExt, Trange(1), ...
            Trange(end), sf);
    end
    if (postprocess_Ca)
        if (postprocess_Glutamate)
            [spineCa, XdimInner, YdimInner, ZdimInner] = ...
                load4D(directory, [glutamate_prep,'SpineCa'], fileExt, Trange(1), ...
                Trange(end), sf);
        end
    end
    if (postprocess_eCB)
        if (~perturbation)
            if (postprocess_Glutamate)
                [~, figNum] = newFigure(figNum, false);
                plotModulation(directory, [glutamate_prep,'SpineeCBproduction'], fileExt, ...
                    'Spine eCB Modulation Function', 'Ca2+', 'eCB+', ...
                    [glutamate_prep,'eCBproduction'], fontsize);
            end
        end
        [spineeCB, XdimInner, YdimInner, ZdimInner] = ...
            load4D(directory, [glutamate_prep,'SpineeCB'], fileExt, Trange(1), ...
            Trange(end), sf);
    end
    if (postprocess_OutputSpikes)
        [outputSpike, Xdim, Ydim, Zdim] = loadSpikes(directory, ...
            'Output_Spikes', fileExt, postprocess_OutputSpikesFilter, ...
            spikeTrange(1), spikeTrange(end), Trange(1), Trange(end), dt);
    end
    % Plot
    if (postprocess_InputSpikes)
        if (postprocess_Glutamate)
            [~, figNum] = newFigure(figNum, false); % Population firing rate histogram over all time
            glutamateInHz = calculateHz(glutamateInputSpike, spikeTrange(1), ...
                spikeTrange(end), HzSf, Xdim, ...
                Ydim, Zdim, dt);
            histogram(mean(glutamateInHz(:),2));
            title(['Glutamate in Hz',perturbationString(perturbation,0,1)]); 
            xlabel('Hz'); ylabel('count');
            set([gca; findall(gca, 'Type','text')], 'FontSize', fontsize);
            print([directory,glutamate_prep,'inHz',perturbationString(perturbation,1,0)],'-dpng');
        end
    end
    synapse=randi(XdimInner,1,1); % only for Xdimension
    [~, figNum] = newFigure(figNum, false);
    glutamateFigNum = figNum;
    if (postprocess_InputSpikes)
        if (postprocess_Glutamate)
            figure(glutamateFigNum);
            subplot(5,4,1); 
            scatter(glutamateInputSpike{glutamatePreIndexs(1,synapse,1,1),1,1}.*dt, ...
                ones(numel(glutamateInputSpike{glutamatePreIndexs(1,synapse,1,1),1,1}),1),'.');
            title(['Glutamate Input Spikes',perturbationString(perturbation,0,1)]); 
            xlim([Trange(1) Trange(end)]);
            set(gca,'XTick',[]);
            set([gca; findall(gca, 'Type','text')], 'FontSize', fontsize);
        end
    end
    if (postprocess_Neurotransmitter || postprocess_AvailableNeurotransmitter)
        if (postprocess_Glutamate)
            figure(glutamateFigNum);
            subplot(5,4,2); hold on;
            if (postprocess_Neurotransmitter)
                plot(sfRange, glutamateNeurotransmitter(:,synapse,1,1));
            end
            if (postprocess_AvailableNeurotransmitter)
                plot(sfRange, glutamateAvailableNeurotransmitter(:,synapse,1,1));
            end
            xlim([Trange(1) Trange(end)]);
            title(['Glutamate and Available Glutamate',perturbationString(perturbation,0,1)]); 
            set(gca,'XTick',[]);
            set([gca; findall(gca, 'Type','text')], 'FontSize', fontsize);
        end
        end
    end
    if (postprocess_CB1R)
        if (postprocess_Glutamate)
            figure(glutamateFigNum);
            subplot(5,4,3);
            plot(sfRange, glutamateCB1R(:,synapse,1,1));
            xlim([Trange(1) Trange(end)]);
            title(['Glutamate CB1R',perturbationString(perturbation,0,1)]); 
            set(gca,'XTick',[]);
            set([gca; findall(gca, 'Type','text')], 'FontSize', fontsize);
        end
    end 
    if (postprocess_CB1Runbound)
        if (postprocess_Glutamate)
            figure(glutamateFigNum);
            subplot(5,4,4);
            plot(sfRange, glutamateCB1Runbound(:,synapse,1,1));
            xlim([Trange(1) Trange(end)]);
            title(['Glutamate CB1R unbound',perturbationString(perturbation,0,1)]); 
            set(gca,'XTick',[]);
            set([gca; findall(gca, 'Type','text')], 'FontSize', fontsize);
        end
    end 
    if (postprocess_CB1Rcurrent)
        if (postprocess_Glutamate)
            figure(glutamateFigNum);
            subplot(5,4,5);
            plot(sfRange, glutamateCB1Rcurrent(:,synapse,1,1));
            xlim([Trange(1) Trange(end)]);
            title(['Glutamate CB1R Current',perturbationString(perturbation,0,1)]); 
            set(gca,'XTick',[]);
            set([gca; findall(gca, 'Type','text')], 'FontSize', fontsize);
        end
    end 
    if (postprocess_GoodwinX)
        subplot(5,4,6);
        plot(sfRange, GoodwinX(:,synapse,1,1));
        xlim([Trange(1) Trange(end)]);
        title(['GoodwinX',perturbationString(perturbation,0,1)]); 
        set(gca,'XTick',[]);
    end 
    if (postprocess_GoodwinY)
        subplot(5,4,7);
        plot(sfRange, GoodwinY(:,synapse,1,1));
        xlim([Trange(1) Trange(end)]);
        title(['GoodwinY',perturbationString(perturbation,0,1)]); 
        set(gca,'XTick',[]);
    end 
    if (postprocess_GoodwinZ)
        subplot(5,4,8);
        plot(sfRange, GoodwinZ(:,synapse,1,1));
        xlim([Trange(1) Trange(end)]);
        title(['GoodwinZ',perturbationString(perturbation,0,1)]); 
        set(gca,'XTick',[]);
    end 
    if (postprocess_CleftAstrocyteNeurotransmitter)
        if (postprocess_Glutamate)
            figure(glutamateFigNum);
            subplot(5,4,9);
            plot(sfRange, glutamateCleftAstrocyteNeurotransmitter(:,synapse,1,1));
            xlim([Trange(1) Trange(end)]);
            title(['Glutamate Cleft',perturbationString(perturbation,0,1)]); 
            set(gca,'XTick',[]);
            set([gca; findall(gca, 'Type','text')], 'FontSize', fontsize);
        end
    end
    if (postprocess_CleftAstrocyteeCB)
        if (postprocess_Glutamate)
            figure(glutamateFigNum);
            subplot(5,4,10);
            plot(sfRange, glutamateCleftAstrocyteeCB(:,synapse,1,1));
            xlim([Trange(1) Trange(end)]);
            title(['Glutamate Cleft eCB',perturbationString(perturbation,0,1)]); 
            set(gca,'XTick',[]);
            set([gca; findall(gca, 'Type','text')], 'FontSize', fontsize);
        end
    end    
    if ((postprocess_AMPAWeights || postprocess_AMPA) && postprocess_Glutamate)
        figure(glutamateFigNum);
        subplot(5,4,11); hold on;
        if (postprocess_AMPAWeights)
            plot(sfRange, ones(1,numel(sfRange))*spineAMPAWeights(synapse,1,1));
            yyaxis right;
        end
        if (postprocess_AMPA)
            plot(sfRange, spineAMPA(:,synapse,1,1));  
        end
        xlim([Trange(1) Trange(end)]);
        title(['Spine AMPA/AMPAWeights',perturbationString(perturbation,0,1)]); 
        set(gca,'XTick',[]);
        set([gca; findall(gca, 'Type','text')], 'FontSize', fontsize);
    end
    if (postprocess_mGluR5 && postprocess_Glutamate)
        figure(glutamateFigNum);
        subplot(5,4,12);
        plot(sfRange, spinemGluR5(:,synapse,1,1));
        xlim([Trange(1) Trange(end)]);
        title(['Spine mGluR5',perturbationString(perturbation,0,1)]); 
        set(gca,'XTick',[]);
        set([gca; findall(gca, 'Type','text')], 'FontSize', fontsize);
    end
    if ((postprocess_NMDARopen || postprocess_NMDARCacurrent) && postprocess_Glutamate)
        figure(glutamateFigNum);
        subplot(5,4,13);
        if (postprocess_NMDARopen)
            plot(sfRange, spinemNMDARopen(:,synapse,1,1));
            yyaxis right;
        end
        if (postprocess_NMDARCacurrent)
            plot(sfRange, spinemNMDARCacurrent(:,synapse,1,1));
        end
        xlim([Trange(1) Trange(end)]);
        title(['Spine NMDARopen/NMDARCacurrent',perturbationString(perturbation,0,1)]);
        set(gca,'XTick',[]);
        set([gca; findall(gca, 'Type','text')], 'FontSize', fontsize);
    end
    if (postprocess_Ca)
        if (postprocess_Glutamate)
            figure(glutamateFigNum);
            subplot(5,4,14);
            plot(sfRange, spineCa(:,synapse,1,1));
            xlim([Trange(1) Trange(end)]);
            title(['Spine Ca',perturbationString(perturbation,0,1)]); 
            set(gca,'XTick',[]);
            set([gca; findall(gca, 'Type','text')], 'FontSize', fontsize);
        end
    end
    if (postprocess_eCB)
        if (postprocess_Glutamate)
            figure(glutamateFigNum);
            subplot(5,4,15);
            plot(sfRange, spineeCB(:,synapse,1,1));
            xlim([Trange(1) Trange(end)]);
            title(['Spine eCB',perturbationString(perturbation,0,1)]); 
            set(gca,'XTick',[]);
            set([gca; findall(gca, 'Type','text')], 'FontSize', fontsize);
        end
    end
    if (postprocess_OutputSpikes)
        if (postprocess_Glutamate)
            figure(glutamateFigNum);
            subplot(5,4,16);
            scatter(outputSpike{glutamatePreIndexs(1,synapse,1,1),1,1}.*dt, ...
                ones(numel(outputSpike{glutamatePreIndexs(1,synapse,1,1),1,1}),1),'.');
            xlim([Trange(1) Trange(end)]);
            title(['Glutamate Output Spike',perturbationString(perturbation,0,1)]); 
            set(gca,'XTick',[]);
            set([gca; findall(gca, 'Type','text')], 'FontSize', fontsize);
        end
    end
    if (postprocess_InputSpikes || postprocess_Neurotransmitter ...
            || postprocess_AvailableNeurotransmitter || postprocess_CB1R ...
            || postprocess_CB1Runbound || postprocess_CB1Rcurrent ...
            || postprocess_CleftNeurotransmitter || postprocess_AMPA ...
            || postprocess_AMPAWeights || postprocess_mGluR5 ...
            || postprocess_Ca || postprocess_eCB ...
            || postprocess_OutputSpikes)
        if (postprocess_Glutamate)
            figure(glutamateFigNum);
            print([directory,glutamate_prep,'Components',perturbationString(perturbation,1,0)],'-dpng');
        end
    end  
    if (postprocess_OutputSpikes)
        [~, figNum] = newFigure(figNum, false); % Population firing rate histogram over all time
        outHz = calculateHz(outputSpike, spikeTrange(1), ...
            spikeTrange(end), HzSf, Xdim, ...
            Ydim, Zdim, dt);
        histogram(mean(outHz(:),2));
        title(['Out Hz',perturbationString(perturbation,0,1)]); 
        xlabel('Hz'); ylabel('count');
        set([gca; findall(gca, 'Type','text')], 'FontSize', fontsize);
        print([directory,'outHz',perturbationString(perturbation,1,0)],'-dpng');
    end
    if (postprocess_mGluR5 && postprocess_Ca && postprocess_eCB && postprocess_Glutamate)
        [~, figNum] = newFigure(figNum, false);
        Hrange = 0:3/50:3;

        spinemGluR5Hist = hist3D(Hrange, spinemGluR5);
        subplot(3,1,1);
        imagesc([0 size(spinemGluR5,1)],Hrange,spinemGluR5Hist);
        title(['Spine mGluR5',perturbationString(perturbation,0,1)]); 
        xlabel('time'); ylabel('mGluR5');
        colormap(flipud(hot)); colorbar(); caxis([0 cbMax]);
        set(gca,'ydir','normal');
        set([gca; findall(gca, 'Type','text')], 'FontSize', fontsize);        

        spineCaHist = hist3D(Hrange, spineCa);
        subplot(3,1,2);
        imagesc([0 size(spineCa,1)],Hrange,spineCaHist);
        title(['Spine Ca2+',perturbationString(perturbation,0,1)]);
        xlabel('time'); ylabel('Ca2+');
        colormap(flipud(hot)); colorbar(); caxis([0 cbMax]);
        set(gca,'ydir','normal');
        set([gca; findall(gca, 'Type','text')], 'FontSize', fontsize);

        spineeCBHist = hist3D(Hrange, spineeCB);
        subplot(3,1,3);
        imagesc([0 size(spineeCB,1)],Hrange,spineeCBHist);
        title(['Spine eCB',perturbationString(perturbation,0,1)]);
        xlabel('time'); ylabel('eCB'); ylim([0 1]);
        colormap(flipud(hot)); colorbar(); caxis([0 cbMax]);
        set(gca,'ydir','normal');
        set([gca; findall(gca, 'Type','text')], 'FontSize', fontsize);
        
        print([directory,glutamate_prep,'spine_mGluR5_Ca_eCB',perturbationString(perturbation,1,0)],'-dpng');
        clear Hrange spinemGluR5Hist spineCaHist spineeCBHist;
    end    
    if (postprocess_CB1R && postprocess_CB1Runbound)
        if (postprocess_Glutamate)
            [~, figNum] = newFigure(figNum, false);
            Hrange = 0:1/50:1;

            glutamateCB1RHist = hist3D(Hrange, glutamateCB1R);
            subplot(2,1,1);
            imagesc([0 size(glutamateCB1R,1)],Hrange,glutamateCB1RHist);
            title(['Glutamate CB1R',perturbationString(perturbation,0,1)]); 
            xlabel('time'); ylabel('CB1R');
            colormap(flipud(hot)); colorbar(); caxis([0 cbMax]);
            set(gca,'ydir','normal');
            set([gca; findall(gca, 'Type','text')], 'FontSize', fontsize);

            glutamateCB1RunboundHist = hist3D(Hrange, glutamateCB1Runbound);
            subplot(2,1,2);
            imagesc([0 size(glutamateCB1Runbound,1)],Hrange,glutamateCB1RunboundHist);
            title(['Glutamate Unbound CB1R',perturbationString(perturbation,0,1)]); 
            xlabel('time'); ylabel('Unbound CB1R');
            colormap(flipud(hot)); colorbar(); caxis([0 cbMax]);
            set(gca,'ydir','normal');
            set([gca; findall(gca, 'Type','text')], 'FontSize', fontsize);
            
            print([directory,glutamate_prep,'CB1R_CB1Runbound',perturbationString(perturbation,1,0)],'-dpng');
            clear Hrange glutamateCB1RHist glutamateCB1RunboundHist; 
        end
        if (postprocess_GABA)
            [~, figNum] = newFigure(figNum, false);
            Hrange = 0:1/50:1;

            GABACB1RHist = hist3D(Hrange, GABACB1R);
            subplot(2,1,1);
            imagesc([0 size(GABACB1R,1)],Hrange,GABACB1RHist);
            title(['GABA CB1R',perturbationString(perturbation,0,1)]); 
            xlabel('time'); ylabel('CB1R');
            colormap(flipud(hot)); colorbar(); caxis([0 cbMax]);
            set(gca,'ydir','normal');
            set([gca; findall(gca, 'Type','text')], 'FontSize', fontsize);

            GABACB1RunboundHist = hist3D(Hrange, GABACB1Runbound);
            subplot(2,1,2);
            imagesc([0 size(GABACB1Runbound,1)],Hrange,GABACB1RunboundHist);
            title(['GABA Unbound CB1R',perturbationString(perturbation,0,1)]); 
            xlabel('time'); ylabel('Unbound CB1R');
            colormap(flipud(hot)); colorbar(); caxis([0 cbMax]);
            set(gca,'ydir','normal');
            set([gca; findall(gca, 'Type','text')], 'FontSize', fontsize);
            
            print([directory,GABA_prep,'CB1R_CB1Runbound',perturbationString(perturbation,1,0)],'-dpng');
            clear Hrange GABACB1RHist GABACB1RunboundHist; 
        end
    end
    if (postprocess_AvailableNeurotransmitter)
        if (postprocess_Glutamate)
            Hrange = 0:0.01:2.1;
            glutamateAvailableNeurotransmitterHist = hist3D(Hrange, ...
                glutamateAvailableNeurotransmitter);
            [~, figNum] = newFigure(figNum, false);
            imagesc([0 size(glutamateAvailableNeurotransmitter,1)]...
                ,Hrange,glutamateAvailableNeurotransmitterHist);
            title(['Glutamate Available Neurotransmitter',perturbationString(perturbation,0,1)]);
            xlabel('time'); 
            ylabel('Available Neurotransmitter');
            colormap(flipud(hot)); colorbar(); caxis([0 cbMax/2]);
            set(gca,'ydir','normal');
            set([gca; findall(gca, 'Type','text')], 'FontSize', fontsize);
            print([directory,glutamate_prep,'availableNeurotransmitter_hist',perturbationString(perturbation,1,0)],'-dpng');
            clear Hrange glutamateAvailableNeurotransmitterHist;
        end
        if (postprocess_GABA)
            Hrange = 0:0.01:2.1;
            GABAAvailableNeurotransmitterHist = hist3D(Hrange, ...
                GABAAvailableNeurotransmitter);
            [~, figNum] = newFigure(figNum, false);
            imagesc([0 size(GABAAvailableNeurotransmitter,1)]...
                ,Hrange,GABAAvailableNeurotransmitterHist);
            title(['GABA Available Neurotransmitter',perturbationString(perturbation,0,1)]);
            xlabel('time'); 
            ylabel('Available Neurotransmitter');
            colormap(flipud(hot)); colorbar(); caxis([0 cbMax/2]);
            set(gca,'ydir','normal');
            set([gca; findall(gca, 'Type','text')], 'FontSize', fontsize);
            print([directory,GABA_prep,'availableNeurotransmitter_hist',perturbationString(perturbation,1,0)],'-dpng');
            clear Hrange GABAAvailableNeurotransmitterHist;
        end
    end
    if (postprocess_Headroom)
        if (postprocess_Glutamate)
            Hrange = -2.1:0.01:2.1;
            [~, figNum] = newFigure(figNum, false);
            glutamateHeadroomH = histHeadroom3D(Hrange, glutamateAvailableNeurotransmitter, ...
                spineAMPAWeights1D);
            imagesc([0 size(glutamateAvailableNeurotransmitter,1)],Hrange,glutamateHeadroomH);
            title(['Glutamate Excess Neurotransmitter',perturbationString(perturbation,0,1)]);
            xlabel('time'); ylabel('excess neurotransmitter');
            colormap(flipud(hot)); colorbar();
            set(gca,'ydir','normal');
            set([gca; findall(gca, 'Type','text')], 'FontSize', fontsize);
            print([directory,glutamate_prep,'excessNeurotransmitter',...
                perturbationString(perturbation,1,0)],'-dpng');
            [~, figNum] = newFigure(figNum, false);
            plot(Hrange,glutamateHeadroomH(:,1));
            title(['Glutamate Excess Neurotransmitter Distribution - Initial Condition',...
                perturbationString(perturbation,0,1)]);
            xlabel('excess neurotransmitter'); ylabel('count');
            xlim([Hrange(1) Hrange(end)]);
            set([gca; findall(gca, 'Type','text')], 'FontSize', fontsize);
            print([directory,glutamate_prep,'excessNeurotransmitterDistInitialCondition'...
                ,perturbationString(perturbation,1,0)],'-dpng');
            [~, figNum] = newFigure(figNum, false);
            plot(Hrange,glutamateHeadroomH(:,end));
            title(['Glutamate Excess Neurotransmitter Distribution - Steady State',...
                perturbationString(perturbation,0,1)]);
            xlabel('excess neurotransmitter'); ylabel('count');
            xlim([Hrange(1) Hrange(end)]);
            set([gca; findall(gca, 'Type','text')], 'FontSize', fontsize);
            print([directory,glutamate_prep,'excessNeurotransmitterDistSteadyState',...
                perturbationString(perturbation,1,0)],'-dpng');
            clear Hrange glutamateHeadroomH;
            
            % correlation of AMPA/Hz with headroom/timeToHeadroom
            [glutamateHeadroomFinal, glutamateHeadroomFinalStd, glutamateHzFinal, glutamateTimeToHeadroom, ...
                glutamateTimeToHeadroomConverged, figNum] = ...
                plotComponentVsHeadroom(figNum, Xdim, Ydim, Zdim, ...
                    XdimInner, YdimInner, ZdimInner, ...
                    glutamateAvailableNeurotransmitter, spineAMPAWeights, ...
                    spineAMPAWeights1D, glutamateInHz, ...
                    glutamatePreIndexs1D, measurementTrange, measurementTrange, ...
                    finalHeadroomStd, Trange(1), sf, HzSf, spikeTrange(1), ...
                    HzMax, AMPAMax, perturbationString(perturbation,0,1), ...
                    [directory,glutamate_prep], perturbationString(perturbation,1,0),...
                    fontsize);
        end
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
            scatterhist(spineAMPAWeights1D(timeToHeadroomConverged), ...
                tempHz(timeToHeadroomConverged), ...
                'Group', riskyGroup(timeToHeadroomConverged), 'Kernel', 'on', 'Parent', hp2, ...
                'Marker','.');
            xlim([0 AMPAMax]); xlabel('glutamate bound to AMPA');
            ylim([0 HzMax]); ylabel('in Hz');
            legend off;

            hp3 = uipanel('position', [0.0 0.0 0.5 0.5]);
            tempCleft = sum(cleftGlutamate((measurementTrange(1)/sf)-(Trange(1)/sf): ...
                (measurementTrange(end)/sf)-(Trange(1)/sf),:,:,:));
            scatterhist(spineAMPAWeights1D(timeToHeadroomConverged), ...
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
                beforePerturbationAMPAWeights1D = spineAMPAWeights1D;
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
                        afterPerturbationAMPAWeights1D = spineAMPAWeights1D;
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
                    if (postprocess_PerturbationHz && postprocess_PerturbationAMPA)
                        perturbationType = 'Both';
                    else
                        continue;
                    end
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
    y, saveFile, fontsize)
    % mGluR5 modulation function only first
    fid = fopen([directory,file,fileExt],'r');
    mGluR5modulation = fread(fid, Inf, 'float');
    fclose(fid);
    clear fid;
    plot(0:1/1000:2, mGluR5modulation);
    title(titleStr);
    xlabel(x); ylabel(y);
    set([gca; findall(gca, 'Type','text')], 'FontSize', fontsize);
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
    directory, additionalSave, fontsize)  
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
    set([gca; findall(gca, 'Type','text')], 'FontSize', fontsize);
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
    set([gca; findall(gca, 'Type','text')], 'FontSize', fontsize);
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
    set([gca; findall(gca, 'Type','text')], 'FontSize', fontsize);
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
    set([gca; findall(gca, 'Type','text')], 'FontSize', fontsize);
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
