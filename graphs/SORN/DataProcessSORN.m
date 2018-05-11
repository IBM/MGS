%% Data processing for SORN

% Connection Weights (xxxxxx denotes the timestep at which saving was done) 

%   E2E_Weights_xxxxxx.dat : Cortico-cortical connections (Feedback loop) [created by ZhengSORNExcUnit]

% Node Temporal Activity 
%   Cortical (XXX denotes interpretation of cortical regions i.e. M1, Msup, S1, Ssec)
%       XXXs.txt : supra-granular (L2/3) activity (up and down states)
%       XXXi.txt : infra-granular (L5) activity (spike times)

%close all;
dt=0.001; 
t_start=300; t_stop=500; % beginning and end chosen for plotting (in sec, from beggining of simulation)
t_wavelets = 30;
sig_length = (t_stop-t_start)/dt;
fileExt = '.dat';

% for connectivity matrix analysis
idx_saved = 0:10^5:3*10^5; %10^6;
idx_saved(1)=1; %idx_saved(end)=2*10^6; idx_saved(end-1)=1*10^6; %idx_saved(end-2)=1*10^6; 

%Cxoffs = [0; 55; 146; 238; 400];
Cxoffs = [0, 400];
Cxsz = [400; 80];
%inpFB = [3, 1, 4, 2];
filepath = '/home/naze/MGS/graphs/SORN/2018may11/stimCfrac0_5/muIP0_1/ratioIP1_0/etaSTDP0_000001/tauSTDP0_9/muHIP0_001/ratioHIP0_5/muHIPi0_005/ratioHIPi0_5/EIratio0_8/E2X0_05/I2X0_1/muDelay1_3_6/ratioDelay0_1/0/';  % your path to IBEx here if needed
%filepath = filepath{1};

% Flags
show_activity = 1;
show_connectivities = 1;
plot_figs = 1 ;
fig_visible = 'on';
save_figs = 1;
save_postprocessing = 1;
saved_binary = 1;
compute_cc = 1;
compute_matrix_permutation = 1;
perform_clustering = 1;
compute_traj = 1;
plot_wavelets = 1;
spiking_stats = 1;
process_TEs =1;

postprocess = {}; % data to be saved at the end of postprocessing 

%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot rasters (SORN)
%%%%%%%%%%%%%%%%%%%%%%%%%
if show_activity
    %% load data
    files = {['SORN-EOutput'], ['SORN-IOutput']};
    var = {['Exc'], ['Inh']};
    if saved_binary
        for f_idx = 1:numel(files)
            fid = fopen(strcat(filepath,files{f_idx},fileExt),'r');
            XdimSORNSpikes = fread(fid, 1, 'int');
            YdimSORNSpikes = fread(fid, 1, 'int');
            temp = fread(fid, Inf, 'int');
            fclose(fid);
            clear fid;
            % 
            temp = reshape(temp, 2, numel(temp)/2)';
            expr = strcat(var(f_idx),'_units = temp(:,1);');
            eval(expr{1});
            expr = strcat(var(f_idx),'_times = temp(:,2);');
            eval(expr{1});
            clear temp;
            expr = strcat(var(f_idx), '_idx = find(',var(f_idx),'_times >= (t_start/dt) & ',var(f_idx),'_times <= (t_stop/dt));');
            eval(expr{1});
        end
    else
        for f_idx = 1:numel(files)
            filename = strcat(filepath, files{f_idx}, fileExt);
            fid = fopen(filename,'r');
            data = textscan(fid,'%f');
            data = data{1};
            fclose(fid);

            % data structure in file:
            % dimX dimY
            % t id 
            % t id
            % ...
            % t in sec; id: unit number in the grid

            dimX = data(1); dimY = data(2); % extract grid size

            expr = strcat(var(f_idx), '_times = data(3:2:end);');
            eval(expr{1});
            expr = strcat(var(f_idx),'_units = data(4:2:end);');
            eval(expr{1});
            clear data;

        %     expr = strcat(var(idx),'_mean =  zeros((t_stop-t_start)/dt+1, 1);');
        %     eval(expr{1});

            expr = strcat(var(f_idx),'_idx = find(',var(f_idx),'_times > (t_start/dt) & ', var(f_idx),'_times < (t_stop/dt));');
            eval(expr{1});

        %     expr = strcat(var(idx),'_idx_start = find(',var(idx),'_times>=t_start);');
        %     eval(expr{1});
        %     expr = strcat(var(idx),'_idx_stop = find(', var(idx), '_times>t_stop);');
        %     eval(expr{1});

            offset = Cxoffs(f_idx)+1;

            expr = strcat('scatter(', var(f_idx),'_times(',var(f_idx),'_idx), ',var(f_idx),'_units(',var(f_idx),'_idx)+offset);');
            eval(expr{1});
            hold on;
        end
    end
    %% Plot figure
    if plot_figs
        figure('Position', [200 200 3200 600], 'Visible', fig_visible);
        for idx = 1:numel(var)
            expr = strcat('area_idx = ',var(idx),'_idx;');    
            eval(expr{1});
            expr = strcat('area_times = ',var(idx),'_times;');    
            eval(expr{1});
            expr = strcat('area_units = ',var(idx),'_units;');    
            eval(expr{1});
            scatter(area_times(area_idx), area_units(area_idx)+Cxoffs(idx)+1);
            hold on;
        end
        xlabel('time(s)'); ylabel('L5 units');
        title('SORN L5 Exc activity');
        xlim([t_start/dt+sig_length/2-1/dt,t_start/dt+sig_length/2+1/dt]);
        if(save_figs)
            pause(2);
            %saveas(gcf, strcat(filepath, 'SORN_scatter_', num2str(round(t_start)), '_', num2str(t_stop), 's_zoom'), 'svg');
            %saveas(gcf, strcat(filepath, 'SORN_scatter_', num2str(round(t_start)), '_', num2str(t_stop), 's_zoom'), 'jpg');
            print(gcf, strcat(filepath, 'SORN_scatter_', num2str(round(t_start)), '_', num2str(t_stop), 's_zoom'), '-depsc');
            print(gcf, strcat(filepath, 'SORN_scatter_', num2str(round(t_start)), '_', num2str(t_stop), 's_zoom'), '-dpng');
        end
%         xlim([t_start/dt t_stop/dt]);
%         if(save_figs)
%             pause(1);
%             saveas(gcf, strcat(filepath, 'SORN_scatter_', num2str(round(t_start)), '_', num2str(t_stop), 's'), 'svg');
%             saveas(gcf, strcat(filepath, 'SORN_scatter_', num2str(round(t_start)), '_', num2str(t_stop), 's'), 'jpg');
%         end
    end
    %% Raster plot
    % re-order indices to get synfire chain
    Exc_sp = sparse(round((Exc_times(Exc_idx)-t_start/dt))+1, Exc_units(Exc_idx)+1, ones(size(Exc_idx)), sig_length+2, Cxsz(1));
    Inh_sp = sparse(round((Inh_times(Inh_idx)-t_start/dt))+1, Inh_units(Inh_idx)+1, ones(size(Inh_idx)), sig_length+2, Cxsz(2));
    if compute_matrix_permutation
        C = cov(Exc_sp(end-100000:end,:)); %take the last 100s of period
        Cs = C > mean(mean(C));
        p = symamd(Cs);
        if plot_figs
            figure('Position', [200 200 3200 600], 'Visible', fig_visible);
            imagesc(Exc_sp(:,p)')
            colormap(flipud(gray));
            xlabel('time(ms)'); ylabel('ordered units');
            if(save_figs)
                pause(1);
                %saveas(gcf, strcat(filepath, 'SORN_ordered_rasters_', num2str(round(t_start)), '_', num2str(t_stop), 's'), 'svg');
                %saveas(gcf, strcat(filepath, 'SORN_ordered_rasters_', num2str(round(t_start)), '_', num2str(t_stop), 's'), 'jpg');
                print(gcf, strcat(filepath, 'SORN_ordered_rasters_', num2str(round(t_start)), '_', num2str(t_stop), 's'), '-depsc');
                print(gcf, strcat(filepath, 'SORN_ordered_rasters_', num2str(round(t_start)), '_', num2str(t_stop), 's'), '-dpng');
                xlim([49500,50500]);
                %saveas(gcf, strcat(filepath, 'SORN_ordered_rasters_', num2str(round(t_start)), '_', num2str(t_stop), 's_zoom'), 'svg');
                %saveas(gcf, strcat(filepath, 'SORN_ordered_rasters_', num2str(round(t_start)), '_', num2str(t_stop), 's_zoom'), 'jpg');
                print(gcf, strcat(filepath, 'SORN_ordered_rasters_', num2str(round(t_start)), '_', num2str(t_stop), 's_zoom'), '-depsc');
                print(gcf, strcat(filepath, 'SORN_ordered_rasters_', num2str(round(t_start)), '_', num2str(t_stop), 's_zoom'), '-dpng');
            end
        end
    end

    %% Cross correlation
    if compute_cc && compute_matrix_permutation
        cc = corrcoef(Exc_sp); % /!\ cc can contain NaNs /!\
        if plot_figs
            figure('Visible', fig_visible); imagesc(cc(p,p));
            xlabel('Ordered units'); ylabel('Ordered units');
            title('Correlation Coefficient');
            colorbar;
            if(save_figs)
                pause(1);
                %saveas(gcf, strcat(filepath, 'SORN_corrCoeff_', num2str(round(t_start)), '_', num2str(t_stop), 's'), 'svg');
                %saveas(gcf, strcat(filepath, 'SORN_corrCoeff_', num2str(round(t_start)), '_', num2str(t_stop), 's'), 'jpg');
                print(gcf, strcat(filepath, 'SORN_corrCoeff_', num2str(round(t_start)), '_', num2str(t_stop), 's'), '-depsc');
                print(gcf, strcat(filepath, 'SORN_corrCoeff_', num2str(round(t_start)), '_', num2str(t_stop), 's'), '-dpng');
            end
        end
        postprocess{end+1} = {'corrCoeff', cc};
    end
    
    %% Clustering
    if plot_figs && perform_clustering
        figure('Visible', fig_visible); imagesc(Cs(p,p));
        if (save_figs) 
            %saveas(gcf, strcat(filepath, 'covMatrix_ordered', num2str(round(t_start)), '_', num2str(t_stop), 's'), 'svg'); 
            %saveas(gcf, strcat(filepath, 'covMatrix_ordered', num2str(round(t_start)), '_', num2str(t_stop), 's'), 'jpg'); 
            print(gcf, strcat(filepath, 'covMatrix_ordered', num2str(round(t_start)), '_', num2str(t_stop), 's'), '-depsc'); 
            print(gcf, strcat(filepath, 'covMatrix_ordered', num2str(round(t_start)), '_', num2str(t_stop), 's'), '-dpng'); 
        end;
        [clustLabel varType] = dbscan(Cs(p,p), 10, 4);
        nb_clusters = numel(unique(clustLabel))
        postprocess{end+1} = {'nbClusters', nb_clusters};
    end
    
    %% Spectrum
    %%%%%%% L5 mean field and spectrum %%%%%%%
    SORN_mean = zeros((t_stop-t_start)/dt+1, size(files,2));
    Pxx = {};
    % create low pass filter
    nth=4; bf=200; sh=1/dt; wf=2*bf./sh;
    [b,a] = butter(nth, wf, 'low');
    for idx = 1:size(files,2)
        % set variable naming framework
        expr = strcat('area_idx = ',var(idx),'_idx;');    
        eval(expr{1});
        expr = strcat('area_times = ',var(idx),'_times;');    
        eval(expr{1});

        % mean popultation activity
        for i = 1:numel(area_idx)
            new_time = round((area_times(area_idx(i))-t_start/dt));
            SORN_mean(new_time+1, idx) = SORN_mean(new_time+1, idx) + 1;
        end
        SORN_mean(:,idx) = SORN_mean(:,idx) / Cxsz(idx);

        % log freq scale
        freqs_log = exp(log(0.1):0.2:log(500));
        Pxx_log{idx} = pmtm(SORN_mean(:,idx), 4, freqs_log, 1/dt);

        % linear freq scale
        freqs_linear = 0.5:0.5:200;
        Pxx{idx} = pmtm(SORN_mean(:,idx), 4, freqs_linear, 1/dt);
    end

    if plot_figs
        SORN_ts_fig = figure('Position', [200 200 3200 300], 'Visible', fig_visible);
        SORN_spect_fig = figure('Position', [3000 100 1200 400], 'Visible', fig_visible);
        for idx = 1:size(files,2)
            % filtered mean
            set(0, 'currentfigure', SORN_ts_fig);
            sig = filtfilt(b,a,SORN_mean(:,idx));
            plot(sig); hold on;

            % log-log plot
            set(0, 'currentfigure', SORN_spect_fig);
            subplot(1,2,2);
            plot(freqs_log, 10*log10(Pxx_log{idx})); hold on;

            % linear-linear plot
            subplot(1,2,1);
            plot(freqs_linear, Pxx{idx}); hold on; 
        end

        set(0, 'currentfigure', SORN_ts_fig);
        xlabel('time(ms)'); ylabel('mean amplitude');
        xlim([0 size(SORN_mean,1)]);
        if(save_figs)
            pause(1);
            %saveas(gcf, strcat(filepath, 'SORN_mean_ts_', num2str(round(t_start)), '_', num2str(t_stop), 's'), 'svg');
            %saveas(gcf, strcat(filepath, 'SORN_mean_ts_', num2str(round(t_start)), '_', num2str(t_stop), 's'), 'jpg');
            print(SORN_ts_fig, strcat(filepath, 'SORN_mean_ts_', num2str(round(t_start)), '_', num2str(t_stop), 's'), '-depsc');
            print(SORN_ts_fig, strcat(filepath, 'SORN_mean_ts_', num2str(round(t_start)), '_', num2str(t_stop), 's'), '-dpng');
            pause(1)
        end
        set(0, 'currentfigure', SORN_ts_fig);
        xlim( [ size(SORN_mean,1)/2-1/dt size(SORN_mean,1)/2+1/dt ] ); % 2 sec around middle of time serie
        if(save_figs)
            pause(1);
            %saveas(gcf, strcat(filepath, 'SORN_mean_ts_', num2str(round(t_start)), '_', num2str(t_stop), 's_zoom'), 'svg');
            %saveas(gcf, strcat(filepath, 'SORN_mean_ts_', num2str(round(t_start)), '_', num2str(t_stop), 's_zoom'), 'jpg');
            print(gcf, strcat(filepath, 'SORN_mean_ts_', num2str(round(t_start)), '_', num2str(t_stop), 's_zoom'), '-depsc');
            print(gcf, strcat(filepath, 'SORN_mean_ts_', num2str(round(t_start)), '_', num2str(t_stop), 's_zoom'), '-dpng');
            pause(1)
        end

        set(0, 'currentfigure', SORN_spect_fig); 
        subplot(1,2,2);
        set(gca, 'xscale', 'log'); xlabel('Frequency (Hz)')
        ylabel('Power (dB)'); %set(gca, 'yscale', 'log');
        legend(var(1:size(files,2)));
        subplot(1,2,1);
        xlabel('Frequency (Hz)')
        ylabel('PSD'); %set(gca, 'yscale', 'log');
        xlim([0 80]);
        if(save_figs)
            pause(1);
            %saveas(gcf, strcat(filepath, 'SORN_spectrum_', num2str(round(t_start)), '_', num2str(t_stop), 's'), 'svg');
            %saveas(gcf, strcat(filepath, 'SORN_spectrum_', num2str(round(t_start)), '_', num2str(t_stop), 's'), 'jpg');
            print(gcf, strcat(filepath, 'SORN_spectrum_', num2str(round(t_start)), '_', num2str(t_stop), 's'), '-depsc');
            print(gcf, strcat(filepath, 'SORN_spectrum_', num2str(round(t_start)), '_', num2str(t_stop), 's'), '-dpng');
        end
    end
    postprocess{end+1} = {'Spectrum_linear', Pxx, freqs_linear};
    postprocess{end+1} = {'Spectrum_log', Pxx_log, freqs_log};
end

%% Wavelets
if plot_wavelets
    mid_time = round(0.5*(t_stop-t_start)/dt);
    wvlts = cwt_morlet_SN(SORN_mean(mid_time:mid_time+t_wavelets/dt,1), 1/dt, 0,freqs_linear, 5, 'gabor', 'no');
    wvlts_fig = figure('Position', [600 600 2400 400], 'Visible', fig_visible);
    wvlts_ax = subplot(1,1,1);
    imagesc(1:dt:t_wavelets, freqs_linear, abs(wvlts));
    set(wvlts_ax, 'YDir', 'Normal');
    ylim([0 100]);
    colormap(jet);
    xlabel('Times (s)'); ylabel('Frequency (Hz)') 
    if save_figs
        pause(1)
        print(gcf, strcat(filepath, 'SORN_wavelets_', num2str(round(t_start)), '_', num2str(t_start+t_wavelets), 's'), '-depsc');
        print(gcf, strcat(filepath, 'SORN_wavelets_', num2str(round(t_start)), '_', num2str(t_start+t_wavelets), 's'), '-dpng');
        pause(1)
        xlim([1 10]);
        print(gcf, strcat(filepath, 'SORN_wavelets_', num2str(round(t_start)), '_', num2str(t_start+t_wavelets), 's_zoom'), '-depsc');
        print(gcf, strcat(filepath, 'SORN_wavelets_', num2str(round(t_start)), '_', num2str(t_start+t_wavelets), 's_zoom'), '-dpng');
    end
    
    
end

%% Spiking stats
if spiking_stats
    exc_spike_times = find(full(Exc_sp)==1);
    inh_spike_times = find(full(Inh_sp)==1);
    
    exc_spike_intervals = diff(exc_spike_times);
    inh_spike_intervals = diff(inh_spike_times);

    isi_fig = figure('Position', [600 600 1600 400], 'Visible', fig_visible);
    ax1 = subplot(1,5,1:2);
    histogram(exc_spike_intervals, exp(1:0.2:10), 'FaceColor', rgb('MediumBlue'));
    xlabel('ISI'); title('Exc Units');
    set(ax1, 'xscale', 'log');

    ax2 = subplot(1,5,3:4);
    histogram(inh_spike_intervals, exp(1:0.2:10), 'FaceColor', rgb('Salmon'));
    xlabel('ISI'); title('Inh Units');
    set(ax2, 'xscale', 'log');

    ax3 = subplot(1,5,5);
    Be = bar(1,std(exc_spike_intervals)/mean(exc_spike_intervals)); %, 'FaceColor', 'b'); %rgb('MediumBlue')); hold on;
    set(Be, 'FaceColor', rgb('MediumBlue'));
    hold on;
    Bi = bar(2,std(inh_spike_intervals)/mean(inh_spike_intervals)); %, 'FaceColor', 'r'); %rgb('Salmon')); hold on;
    set(Bi, 'FaceColor', rgb('Salmon'));
    xlim([0 3]); xticks([1 2]);
    set(ax3, 'XTickLabels', ['Exc'; 'Inh';]);
    title('CV ISI');
    
    if save_figs
        pause(1);
        print(gcf, strcat(filepath, 'SORN_spikingStats_', num2str(round(t_start)), '_', num2str(t_start+t_wavelets), 's'), '-depsc');
        print(gcf, strcat(filepath, 'SORN_spikingStats_', num2str(round(t_start)), '_', num2str(t_start+t_wavelets), 's'), '-dpng');
    end
end

%% Trajectories
if compute_traj
    tauZ = 100; t_artifact = 0.5;
    gauss_conv = normpdf(-round(tauZ/2):round(tauZ/2), 0, 20);  % gaussian kernel to convolute signal (based on tauZ=100)
    svd_ts = full(Exc_sp((10-t_artifact)/dt:(20+t_artifact)/dt,:));
    for ts = 1:(size(svd_ts,1)-tauZ)
        svd_ts(ts,:) = gauss_conv * svd_ts(ts:(ts+tauZ), :);
    end
    svd_ts = svd_ts(t_artifact/dt:end-t_artifact/dt, :);

    comp = [1 2 3];  % which eigenvectors to project 
    [U, S, V] = svd(svd_ts);
    svd_fig = figure('Position', [300 300 600 1200], 'Visible', fig_visible); 

    % eigenvalues histogram
    eig_ax = subplot(3,1,1);
    eigenval = diag(S);
    eigenval = eigenval/sum(eigenval);
    bar(eigenval);
    xlim([1 100]);
    xlabel('eigen index'); ylabel('eigenvalues')

    % trajectories
    traj_ax = subplot(3,1,[2 3]);
    plot3(U(1:2500, comp(1)), U(1:2500, comp(2)), U(1:2500, comp(3)), '.k'); hold on;
    plot3(U(2501:5000, comp(1)), U(2501:5000, comp(2)), U(2501:5000, comp(3)), '.r'); hold on;
    plot3(U(5001:7500, comp(1)), U(5001:7500, comp(2)), U(5001:7500, comp(3)), '.g'); hold on;
    plot3(U(7501:10000, comp(1)), U(7501:10000, comp(2)), U(7501:10000, comp(3)), '.b'); hold on;

    if(save_figs)
        pause(2);
        %saveas(gcf, strcat(filepath, 'SORN_all_SVD_', num2str(round(t_start+10)), '_', num2str(t_start+20), 's'), 'svg');
        %saveas(gcf, strcat(filepath, 'SORN_all_SVD_', num2str(round(t_start+10)), '_', num2str(t_start+20), 's'), 'jpg');
        print(gcf, strcat(filepath, 'SORN_all_SVD_', num2str(round(t_start+10)), '_', num2str(t_start+20), 's'), '-depsc');
        print(gcf, strcat(filepath, 'SORN_all_SVD_', num2str(round(t_start+10)), '_', num2str(t_start+20), 's'), '-dpng');
    end

    entropy = -sum(eigenval .* log(eigenval));
    postprocess{end+1} = {'entropy', entropy};
end

%%
if show_connectivities
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% SORN E2E weight matrices 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    w_conn = zeros(Cxoffs(end), Cxoffs(end), numel(idx_saved));
    c_frac = zeros(length(Cxsz), length(idx_saved)-1);
    normW = zeros(length(Cxsz), length(idx_saved)-1);
    G = {}; % container for graphs
    
    for i=1:length(idx_saved)
        filename = sprintf('E2E_Weights_%i.dat',idx_saved(i));
        x = load ([filepath filename]);
        x = spconvert(x);
        w_conn(1:size(x,1), 1:size(x,2), i) = x;
        G{i} = digraph(w_conn(:,:,i));
        
        % Weight evolution
        for m=1
            normW(m,i) = norm(w_conn(:,:,i), 'fro');
            c_frac(m,i) = sum(sum(w_conn(:,:,i)>0));
        end
        %figure; imagesc(w_conn(Cxoffs(m)+1:Cxoffs(m+1), Cxoffs(m)+1:Cxoffs(m+1)));
        
    end
    
    if plot_figs
        c_fig = figure('Position', [400, 400, 3800, 600], 'Visible', fig_visible, 'name', 'SORN E2E Connectivity Matrix'); % connectivity figure
        wd_fig = figure('Position', [400, 400, 3800, 400], 'Visible', fig_visible, 'name', 'SORN E2E Weights Distribution'); % weight distribution figure
        we_fig  = figure('Position', [400, 400, 1600, 400], 'Visible', fig_visible, 'name', 'SORN E2E Weights Evolution'); % weight evolution figure

        for i = 1:numel(idx_saved)
            
            % Connectivity matrix
            set(0,'currentfigure', c_fig);
            subplot(2,numel(idx_saved),i);
            h=imagesc(w_conn(:,:,i));
            xlabel('Excitatory units');
            ylabel('Excitatory units');
            title(sprintf('t=%ims',idx_saved(i)));
            
            % Graph visualization
            subplot(2,numel(idx_saved),numel(idx_saved)+i);
            plot(G{i});

            % Weight distribution
            set(0, 'currentfigure', wd_fig);
            for j=1
                subplot(1,numel(idx_saved),10*(j-1)+i);
                histogram(w_conn(:,:,i), 'Normalization', 'probability', 'BinMethod', 'scott');
                xlim([0.0001 0.1]);
                ylim([0 0.01]);
                xlabel('weight'); ylabel('probability')
            end
            xlabel(sprintf('t=%ims', idx_saved(i)));
        end

        set(0, 'currentfigure', we_fig);
        for m = 1
            subplot(1,3,1); plot(normW(m,:)); hold on; 
            subplot(1,3,2); plot(diff(normW(m,:))); hold on;
            subplot(1,3,3); plot(c_frac(m,:)/(Cxsz(m)^2)); hold on;
        end
        subplot(1,3,1); xlabel('time (x10^{5})'); ylabel('D_{wee}');
        subplot(1,3,2); xlabel('time (x10^{5})'); ylabel('dD_{wee}');
        subplot(1,3,3); plot(sum(c_frac,1)/(Cxoffs(end)^2), 'k', 'LineWidth', 2); hold on; 
                        xlabel('time (x10^{5})'); ylabel('connection fraction per aera (total in black)');

        if(save_figs)
            pause(1);
%             saveas(c_fig, strcat(filepath, 'SORN_E2E_conn_', num2str(round(t_start)), '_', num2str(t_stop), 's'), 'svg');
%             saveas(wd_fig, strcat(filepath, 'SORN_E2E_distrib_', num2str(round(t_start)), '_', num2str(t_stop), 's'), 'svg');
%             saveas(we_fig, strcat(filepath, 'SORN_E2E_evol_',num2str(round(t_start)), '_', num2str(t_stop), 's'), 'svg');
%             
%             saveas(c_fig, strcat(filepath, 'SORN_E2E_conn_', num2str(round(t_start)), '_', num2str(t_stop), 's'), 'jpg');
%             saveas(wd_fig, strcat(filepath, 'SORN_E2E_distrib_', num2str(round(t_start)), '_', num2str(t_stop), 's'), 'jpg');
%             saveas(we_fig, strcat(filepath, 'SORN_E2E_evol_',num2str(round(t_start)), '_', num2str(t_stop), 's'), 'jpg');
             
            print(c_fig, strcat(filepath, 'SORN_E2E_conn_', num2str(round(t_start)), '_', num2str(t_stop), 's'), '-depsc');
            print(wd_fig, strcat(filepath, 'SORN_E2E_distrib_', num2str(round(t_start)), '_', num2str(t_stop), 's'), '-depsc');
            print(we_fig, strcat(filepath, 'SORN_E2E_evol_',num2str(round(t_start)), '_', num2str(t_stop), 's'), '-depsc');
            
            print(c_fig, strcat(filepath, 'SORN_E2E_conn_', num2str(round(t_start)), '_', num2str(t_stop), 's'), '-dpng');
            print(wd_fig, strcat(filepath, 'SORN_E2E_distrib_', num2str(round(t_start)), '_', num2str(t_stop), 's'), '-dpng');
            print(we_fig, strcat(filepath, 'SORN_E2E_evol_',num2str(round(t_start)), '_', num2str(t_stop), 's'), '-dpng');
        end
        figure('Visible', fig_visible); 
        imagesc(w_conn(p,p,end));
        title('Ordered E2E weights (after clustering)');
        if (save_figs) 
            %saveas(gcf, strcat(filepath, 'SORN_E2E_conn_ordered', num2str(round(t_start)), '_', num2str(t_stop), 's'), 'svg'); 
            %saveas(gcf, strcat(filepath, 'SORN_E2E_conn_ordered', num2str(round(t_start)), '_', num2str(t_stop), 's'), 'jpg'); 
            print(gcf, strcat(filepath, 'SORN_E2E_conn_ordered', num2str(round(t_start)), '_', num2str(t_stop), 's'), '-depsc'); 
            print(gcf, strcat(filepath, 'SORN_E2E_conn_ordered', num2str(round(t_start)), '_', num2str(t_stop), 's'), '-dpng'); 
        end
    end
    
    %% in/out degree
    for i=1:numel(idx_saved)
        in_mean = mean(sum(w_conn(:,:,i)>0,2))/Cxsz(1); 
        out_mean = mean(sum(w_conn(:,:,i)>0,1))/Cxsz(1);
        in_std = std(sum(w_conn(:,:,i)>0,2))/Cxsz(1); 
        out_std = std(sum(w_conn(:,:,i)>0,1))/Cxsz(1); 
    end
    if plot_figs
        figure('Position', [600 600 1200 200], 'Visible', fig_visible);
        for i=1:numel(idx_saved)
            subplot(1,numel(idx_saved),i);
            plot(1, in_mean, '.k', 'MarkerSize', 15); hold on;
            errorbar(1, in_mean, in_std, 'k', 'LineWidth', 2); hold on;
            plot(2, out_mean, '.k', 'MarkerSize', 15); hold on;
            errorbar(2, out_mean, out_std, 'k', 'LineWidth', 2); hold on;
            set(gca, 'XTick', [1 2]); set(gca, 'XTickLabel', {['in'], ['out']});
            ylim([0 0.2]); xlim([0.6 2.4]);
            title(strcat('t=', num2str(idx_saved(i))));
            if (i==1) ylabel('connection fraction (%)'); end;
        end
        if(save_figs)
            pause(1);
            %saveas(gcf, strcat(filepath, 'E2E_degree_', num2str(round(t_start)), '_', num2str(t_stop), 'ms'), 'svg');
            %saveas(gcf, strcat(filepath, 'E2E_degree_', num2str(round(t_start)), '_', num2str(t_stop), 'ms'), 'jpg');
            print(gcf, strcat(filepath, 'E2E_degree_', num2str(round(t_start)), '_', num2str(t_stop), 'ms'), '-depsc');
            print(gcf, strcat(filepath, 'E2E_degree_', num2str(round(t_start)), '_', num2str(t_stop), 'ms'), '-dpng');
        end
    end
    postprocess{end+1} = {'mean_in_out_degrees', [in_mean, out_mean]};

    %% Delay matrix
    w_delays = zeros(Cxoffs(end), Cxoffs(end));
    filename = 'E2E_Delays_1.dat';
    x = load ([filepath filename]);
    x = spconvert(x);
    w_delays(1:size(x,1), 1:size(x,2)) = x;
    
    if plot_figs
        delays_fig = figure('Position', [600, 600, 1000, 500], 'Visible', fig_visible, 'name', 'SORN E2E Delay Matrix');
        subplot(1,5,1:3);
        h=imagesc(w_delays);
        xlabel('Excitatory units');
        ylabel('Excitatory units');
        title(sprintf('Delay Matrix'));
        colorbar;
        subplot(1,5,4:5);
        histogram(w_delays(find(w_delays)), 'Normalization', 'Probability');
        xlabel('Delays'); ylabel('Probability')
        title('Distribution of delays');
    end
    if(save_figs)
            pause(1);
            saveas(gcf, strcat(filepath, 'SORN_E2E_delays_', num2str(round(t_start)), '_', num2str(t_stop), 's'), 'svg');
            saveas(gcf, strcat(filepath, 'SORN_E2E_delays_', num2str(round(t_start)), '_', num2str(t_stop), 's'), 'png');
            %print(gcf, strcat(filepath, 'SORN_E2E_delays_', num2str(round(t_start)), '_', num2str(t_stop), 's'), '-depsc');
            %print(gcf, strcat(filepath, 'SORN_E2E_delays_', num2str(round(t_start)), '_', num2str(t_stop), 's'), '-dpng');
        end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Plot SORN E2I weight matrices 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %idx_saved = 0:10^5:6*10^5;
    %idx_saved(1)=1; idx_saved(end)=10^6;
    if plot_figs
        figure('name', 'SORN E2I Connectivity Matrix', 'Position', [200 200 3800 400], 'Visible', fig_visible);
        for i=1:numel(idx_saved)
            filename = sprintf('E2I_Weights_%i.dat',idx_saved(i));
            x = load ([filepath filename]);
            x = spconvert(x);
            subplot(1,numel(idx_saved),i);
            h=imagesc(x);
            xlabel('Excitatory units');
            ylabel('Inhibitory units');
            title(sprintf('t=%ims',idx_saved(i)));
        end
        if(save_figs)
            pause(1);
            %saveas(gcf, strcat(filepath, 'SORN_E2I_conn_', num2str(round(t_start)), '_', num2str(t_stop), 's'), 'svg');
            %saveas(gcf, strcat(filepath, 'SORN_E2I_conn_', num2str(round(t_start)), '_', num2str(t_stop), 's'), 'jpg');
            print(gcf, strcat(filepath, 'SORN_E2I_conn_', num2str(round(t_start)), '_', num2str(t_stop), 's'), '-depsc');
            print(gcf, strcat(filepath, 'SORN_E2I_conn_', num2str(round(t_start)), '_', num2str(t_stop), 's'), '-dpng');
        end
    end
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Plot SORN I2E weight matrices 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %idx_saved = 0:10^5:6*10^5;
    %idx_saved(1)=1; idx_saved(end)=10^6;
    if plot_figs
        w_conn = zeros(Cxoffs(end), 80);
        figure('Position', [100, 400, 3800, 400], 'Visible', fig_visible, 'name', 'SORN I2E Connectivity Matrix');
        for i=1:length(idx_saved)
            filename = sprintf('I2E_Weights_%i.dat',idx_saved(i));
            x = load ([filepath filename]);
            x = spconvert(x);
            w_conn(1:size(x,1), 1:size(x,2)) = x;
            subplot(1,numel(idx_saved),i);
            h=imagesc(w_conn);
            xlabel('Inhibitory units');
            ylabel('Excitatory units');
            title(sprintf('t=%ims',idx_saved(i)));
        end
        if(save_figs)
            pause(1);
            %saveas(gcf, strcat(filepath, 'SORN_I2E_conn_', num2str(round(t_start)), '_', num2str(t_stop), 's'), 'svg');
            %saveas(gcf, strcat(filepath, 'SORN_I2E_conn_', num2str(round(t_start)), '_', num2str(t_stop), 's'), 'jpg');
            print(gcf, strcat(filepath, 'SORN_I2E_conn_', num2str(round(t_start)), '_', num2str(t_stop), 's'), '-depsc');
            print(gcf, strcat(filepath, 'SORN_I2E_conn_', num2str(round(t_start)), '_', num2str(t_stop), 's'), '-dpng');
        end
    end
end

if process_TEs
    if saved_binary
        fid = fopen(strcat(filepath,'SORN_Exc_TEs',fileExt),'r');
        XdimSORN_TEs = fread(fid, 1, 'int');
        YdimSORN_TEs = fread(fid, 1, 'int');
        TEs = fread(fid, Inf, 'double');
        fclose(fid);
        clear fid;
        % 
        sz = numel(TEs)/(XdimSORN_TEs*YdimSORN_TEs);
        if XdimSORN_TEs==1
            TEs = reshape(TEs, [YdimSORN_TEs, sz]);
        else
            TEs = reshape(TEs, [XdimSORN_TEs, YdimSORN_TEs, sz]);
        end
    else
        disp('Visualization of TEs not implemented when saved in plain text, sorry. Save your simulation output in binary.');
    end
    %%
    if plot_figs
        TEs_fig = figure('Position', [200 200 3200 300], 'Visible', fig_visible);
        ts = 1:sz;
        mu_TEs = mean(TEs);
        sigma_TEs = std(TEs);
        %fill([ts, fliplr(ts)], ([mu_TEs - sigma_TEs/2, fliplr(mu_TEs + sigma_TEs/2)]), 'b', 'EdgeColor', 'none', 'FaceAlpha', 0.002); hold on;
        plot(ts, mu_TEs, '-b', 'LineWidth', 2); hold on;
        plot(ts, mu_TEs + sigma_TEs, '-r', 'LineWidth', 1); hold on;
        plot(ts, mu_TEs - sigma_TEs, '-r', 'LineWidth', 1); hold on;
        xlabel('time (ms)');
        ylabel('Threshold');
    end
end

if save_postprocessing
    save(fullfile(filepath, 'postprocessed_features.mat'), 'postprocess');
end






















