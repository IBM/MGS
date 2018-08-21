%% Data processing for IBEx

% Connection Weights (xxxxxx denotes the timestep at which saving was done) 
%   IMAX_TH_Weights_xxxxxx.dat : Thalamo-cortical projections [created by LinskerUnit]
%   IMAX_LN_Weights_xxxxxx.dat : INFOMAX to SORN projections [created by LinskerUnit]
%   MSN_LN_Weights_xxxxxx.dat : Rabinovich lateral network [created by RabinovichUnit]
%   Cx2Str_Weights_xxxxxx.dat : Cortico-striatal synaptic weights [created by RabinovichUnit]
%   E2E_Weights_xxxxxx.dat : Cortico-cortical connections (Feedback loop) [created by ZhengSORNExcUnit]
%   SN_Weights_xxxxxx.dat : Striato-nigral connections [created by MihalasNieburIAFUnit]
%   PH_Weights_xxxxxx.dat : Phenotypic input [created by GatedThalamoCorticalUnit]
%   DA2Str_Weights_xxxxxx.dat : Dopamine neurons (SNc) projections to Striatum (modulating) [created by RabinovbichWinnerlessUnit]

% Node Temporal Activity 
%   Cortical (XXX denotes interpretation of cortical regions i.e. M1, Msup, S1, Ssec)
%       XXXs.txt : supra-granular (L2/3) activity (up and down states)
%       XXXi.txt : infra-granular (L5) activity (spike times)
%   Thalamic (YYY denotes interpretation of thalamic nuclei i.e. VL, VA, VPL, VPN)
%       YYY.txt : mean field (can be interpreted as LFP) of
%       thalamic units
%       TRN.txt : spike timing of thalamic relay neurons 
%   Striatum
%       D1.txt and D2.txt : MSN activity (interpreted as 1st derivative of
%       calcium transient dCa/dt), labeled D1 and D2 according to their
%       positive vs negative gating on thalamic units
%   SNc (see Mihalas & Niebur (2009) for details of model)
%       Spike_DA.txt : spike timing of dopamine neurons
%       Voltage_DA.txt : sub-threshold activity
%       Threshold_DA.txt : Threshold evolution (activity dependent)


t_start=5500; t_stop=5510; % beginning and end chosen for plotting (in sec, from beggining of simulation)
dt=0.001; 
Cxoffs = [0; 55; 146; 238; 400];
Cxsz = [55; 91; 92; 162];
inpFB = [3, 1 4, 2];
filepath = '/home/naze/nts/graphs/IBEx/traces_may08/';  % your path to IBEx here if needed
figure('Position', [200 200 900 1200]);
show_connectivities=1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot L2/3 activity (Infomax)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
files = {['M1s.txt'], ['Msups.txt'], ['S1s.txt'], ['Ssecs.txt']};
var = {['M1s'], ['Msups'], ['S1s'], ['Ssecs']};
for idx = 1:size(files,2)
    filename = strcat(filepath, files{idx});
    fid = fopen(filename,'r');
    data = textscan(fid,'%f');
    data = data{1};
    fclose(fid);

    % data structure in file:
    % dimX dimY
    % t0  
    % val01 val02 val03 val04 val05 val06 ...
    % 
    % t1  
    % val01 val02 val03 val04 val05 val06 ...
    % ...
    % t in timesteps ; val: value of unit (0.5:up; 0:down)
    
    dimX = data(1); dimY = data(2); % extract grid size; dimX:row dimY:col
    n=dimX*dimY; % nbr of units
    offset=3;
    start_saving_time = data(offset);
    diff_saving_tstart = t_start - start_saving_time;
    %expr = strcat(var(idx), '_times = data(offset+diff_start_saving_time*(n+1):n+1:end);');
    expr = strcat(var(idx), '_times = t_start:dt:t_stop;');
    eval(expr{1});
    expr = strcat(var(idx), '_vals = zeros((t_stop-t_start)/dt, dimX, dimY);');
    eval(expr{1});
    
    offset=offset+1; 
    j=1;
    for i = offset+(t_start/dt - start_saving_time)*(n+1):n+1:offset+(t_stop/dt - start_saving_time)*(n+1)
        expr = strcat(var(idx),'_vals(j,:,:) = reshape(data(i:i+n-1), [dimX, dimY]);');
        eval(expr{1});
        j=j+1;
    end
    clear data;
end
% pick a hundred units to display, 25 from each regions
summary = zeros(100, (t_stop-t_start)/dt+1); 
summary(1:25,:) = permute(M1s_vals(:,:,1:25), [3 1 2]);
summary(26:50,:) = permute(Msups_vals(:,:,1:25), [3 1 2]);
summary(51:75,:) = permute(S1s_vals(:,:,1:25), [3 1 2]);
summary(76:100, :) = permute(Ssecs_vals(:,:,1:25), [3 1 2]);

%subplot(5,1,1);
imagesc(summary);
colormap(flipud(gray));
ylabel('Units');
title('Infomax (L2/3) activity (up & down states)');

%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot L5 rasters (SORN)
%%%%%%%%%%%%%%%%%%%%%%%%%
%subplot(5,1,2);
figure;
files = {['M1i.txt'], ['Msupi.txt'], ['S1i.txt'], ['Sseci.txt'], ['TRN.txt']};
var = {['M1'], ['Msup'], ['S1'], ['Ssec'], ['TRN']};
for idx = 1:size(files,2)
    filename = strcat(filepath, files{idx});
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
    
    expr = strcat(var(idx), '_times = data(3:2:end);');
    eval(expr{1});
    expr = strcat(var(idx),'_units = data(4:2:end);');
    eval(expr{1});
    clear data;
    expr = strcat(var(idx),'_idx_start = find(',var(idx),'_times>=t_start+4);');
    eval(expr{1});
    expr = strcat(var(idx),'_idx_stop = find(', var(idx), '_times>=t_start+6);');
    eval(expr{1});
    
    offset = Cxoffs(idx)+1;
    expr = strcat('scatter(', var(idx),'_times(',var(idx),'_idx_start:',var(idx),'_idx_stop), ',var(idx),'_units(',var(idx),'_idx_start:',var(idx),'_idx_stop)+offset);');
    eval(expr{1});
    hold on;
end
xlim([t_start+4.6,t_start+5.2]);
xlabel('time(s)'); ylabel('L5 units');
title('SORN L5 activity (zoomed scatter, blue:M1; red:Msup; yellow:S1; purple:Ssec; green:TRN/inh)');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot L4/Thalamic activity (GTCU)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
files = {['VL.txt'], ['VA.txt'], ['VPL.txt'], ['VPN.txt']};
var = {['VL'], ['VA'], ['VPL'], ['VPN']};
for idx = 1:size(files,2)
    filename = strcat(filepath, files{idx});
    fid = fopen(filename,'r');
    data = textscan(fid,'%f');
    data = data{1};
    fclose(fid);

    % data structure in file:
    % dimX dimY
    % t0  
    % val01 val02 val03 val04 val05 val06 ...
    % 
    % t1  
    % val01 val02 val03 val04 val05 val06 ...
    % ...
    % t in timesteps ; val: value of unit (0.5:up; 0:down)
    
    dimX = data(1); dimY = data(2); % extract grid size; dimX:row dimY:col
    n=dimX*dimY; % nbr of units
    offset=3;
    expr = strcat(var(idx), '_times = data(offset:n+1:end);');
    eval(expr{1});
    expr = strcat(var(idx), '_vals = zeros((t_stop-t_start)/dt, dimX, dimY);');
    eval(expr{1});
    
    offset=offset+1;
    j=1;
    for i = offset+(t_start/dt)*(n+1):n+1:offset+(t_stop/dt)*(n+1)
        expr = strcat(var(idx),'_vals(j,:,:) = reshape(data(i:i+n-1), [dimX, dimY]);');
        eval(expr{1});
        j=j+1;
    end
    clear data;
end
figure;
%subplot(5,1,3);
plot(mean(VL_vals,3)); hold on;
plot(mean(VA_vals,3)); hold on;
plot(mean(VPL_vals,3)); hold on;
plot(mean(VPN_vals,3)); hold on;
xlim([0,(t_stop-t_start)/dt]);
xlabel('time (ms)'); ylabel('LFP (mean field, filtered)');
title('Thalamic activity');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot Sensory Thalamic activity (Receptive Fields)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
files = {['VLRF.txt'], ['VARF.txt'], ['VPLRF.txt'], ['auditoryRF.txt']};
var = {['VLRF'], ['VARF'], ['VPLRF'], ['VPNRF']};
for idx = 1:size(files,2)
    filename = strcat(filepath, files{idx});
    fid = fopen(filename,'r');
    data = textscan(fid,'%f');
    data = data{1};
    fclose(fid);

    % data structure in file:
    % dimX dimY
    % t0  
    % val01 val02 val03 val04 val05 val06 ...
    % 
    % t1  
    % val01 val02 val03 val04 val05 val06 ...
    % ...
    % t in timesteps ; val: value of unit (0.5:up; 0:down)
    
    dimX = data(1); dimY = data(2); % extract grid size; dimX:row dimY:col
    n=dimX*dimY; % nbr of units
    offset=3;
    expr = strcat(var(idx), '_times = data(offset:n+1:end);');
    eval(expr{1});
    expr = strcat(var(idx), '_vals = zeros((t_stop-t_start)/dt, dimX, dimY);');
    eval(expr{1});
    
    offset=offset+1;
    j=1;
    for i = offset+(t_start/dt)*(n+1):n+1:offset+(t_stop/dt)*(n+1)
        expr = strcat(var(idx),'_vals(j,:,:) = reshape(data(i:i+n-1), [dimX, dimY]);');
        eval(expr{1});
        j=j+1;
    end
    clear data;
end
figure;
subplot(2,1,1);
imagesc(permute(VPNRF_vals, [3 1 2]));
subplot(2,1,2);
for t=5100000:510100
    plot(permute(VPNRF_vals(t,1,:), [3 1 2])); hold on;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot Striatal activity (Rabinovich)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
files = {['D1.txt'], ['D2.txt']};
var = {['D1'], ['D2']};
for idx = 1:size(files,2)
    filename = strcat(filepath, files{idx});
    fid = fopen(filename,'r');
    data = textscan(fid,'%f');
    data = data{1};
    fclose(fid);

    % data structure in file:
    % dimX dimY
    % t0  
    % val01 val02 val03 val04 val05 val06 ...
    % 
    % t1  
    % val01 val02 val03 val04 val05 val06 ...
    % ...
    % t in timesteps ; val: value of unit (0.5:up; 0:down)
    
    dimX = data(1); dimY = data(2); % extract grid size; dimX:row dimY:col
    n=dimX*dimY; % nbr of units
    offset=3;
    expr = strcat(var(idx), '_times = data(offset:n+1:end);');
    eval(expr{1});
    expr = strcat(var(idx), '_vals = zeros((t_stop-t_start)/dt, dimX, dimY);');
    eval(expr{1});
    
    offset=offset+1;
    j=1;
    for i = offset+(t_start/dt)*(n+1):n+1:offset+(t_stop/dt)*(n+1)
        expr = strcat(var(idx),'_vals(j,:,:) = reshape(data(i:i+n-1), [dimX, dimY]);');
        eval(expr{1});
        j=j+1;
    end
    clear data;
end
figure;
%subplot(5,1,4);
str = [D1_vals, D2_vals];
str = reshape(str, [size(str,1), size(D1_vals,3) + size(D2_vals,3)]);
imagesc(str'); colormap(flipud(gray));
xlabel('time(ms)'); ylabel('units');
title('Striatal activity (raster)');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot SNc activity (Mihalas-Niebur)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
files = {['Voltage_DA.txt']};
var = {['DA']};
for idx = 1:size(files,2)
    filename = strcat(filepath, files{idx});
    fid = fopen(filename,'r');
    data = textscan(fid,'%f');
    data = data{1};
    fclose(fid);

    % data structure in file:
    % dimX dimY
    % t0  
    % val01 val02 val03 val04 val05 val06 ...
    % 
    % t1  
    % val01 val02 val03 val04 val05 val06 ...
    % ...
    % t in timesteps ; val: value of unit (0.5:up; 0:down)
    
    dimX = data(1); dimY = data(2); % extract grid size; dimX:row dimY:col
    n=dimX*dimY; % nbr of units
    offset=3;
    expr = strcat(var(idx), '_times = data(offset:n+1:end);');
    eval(expr{1});
    expr = strcat(var(idx), '_vals = zeros((t_stop-t_start)/dt, dimX, dimY);');
    eval(expr{1});
    
    offset=offset+1;
    j=1;
    for i = offset+(t_start/dt)*(n+1):n+1:offset+(t_stop/dt)*(n+1)
        expr = strcat(var(idx),'_vals(j,:,:) = reshape(data(i:i+n-1), [dimX, dimY]);');
        eval(expr{1});
        j=j+1;
    end
    clear data;
end
% plot 10 SNc neurons (supra-threshold spike is here only for vizualisation (i.e. IAF neuron), for exact spike time see Spikes_DA.txt)
figure;
%subplot(5,1,5);
DA_vals(DA_vals>-0.06)=2; 
for i=1:10%size(DA_vals,3)
    plot(2*i+DA_vals(:,:,i)); hold on;
end
xlim([0,(t_stop-t_start)/dt]);
xlabel('time(ms)'); ylabel('membrane potential (shifted, a.u.)');
title('Dopamine neurons (SNc)');

if show_connectivities
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Plot SORN E2E weight matrices 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    idx_saved = 0:10^6:6*10^6;
    idx_saved(1)=1;
    c_frac = zeros(length(Cxsz), length(idx_saved)-1);
    normW = zeros(length(Cxsz), length(idx_saved)-1);
    c_fig = figure('Position', [400, 400, 1200, 600], 'name', 'SORN E2E Connectivity Matrix'); % connectivity figure
    wd_fig = figure('Position', [400, 400, 1200, 600], 'name', 'SORN E2E Weights Distribution'); % weight distribution figure
    we_fig  = figure('Position', [400, 400, 1200, 600], 'name', 'SORN E2E Weights Evolution'); % weight evolution figure
    for i=1:length(idx_saved)-1
        filename = sprintf('E2E_Weights_%i.dat',idx_saved(i));
        x = load ([filepath filename]);
        x = spconvert(x);
        
        % Connectivity matrix
        figure(c_fig);
        subplot(2,5,i);
        p=imagesc(x);
        xlabel('Excitatory units');
        ylabel('Excitatory units');
        title(sprintf('t=%ims',idx_saved(i)));
        
        % Weight distribution
%         figure(wd_fig);
%         for j=1:4
%             subplot(4,10,10*(j-1)+i);
%             histogram(x(Cxoffs(j)+1:Cxoffs(j+1),Cxoffs(inpFB(j))+1:Cxoffs(inpFB(j)+1)), 'Normalization', 'probability', 'BinMethod', 'scott');
%             xlim([0.0001 0.3]);
%             ylim([0 0.1]);
%         end
%         xlabel(sprintf('t=%ims', idx_saved(i)));
        
        % Weight evolution
%         figure(we_fig);
%         for m=1:length(Cxsz)
%             normW(m,i) = norm(x(Cxoffs(m)+1:Cxoffs(m+1), Cxoffs(inpFB(m))+1:Cxoffs(inpFB(m)+1)), 'fro');
%             c_frac(m,i) = sum(sum(x(Cxoffs(m)+1:Cxoffs(m+1), Cxoffs(inpFB(m))+1:Cxoffs(inpFB(m)+1))>0));
%         end
        %figure; imagesc(x(Cxoffs(m)+1:Cxoffs(m+1), Cxoffs(m)+1:Cxoffs(m+1)));
        
    end
    figure(we_fig);
    for m = 1:length(Cxsz)
        subplot(1,3,1); plot(normW(m,:)); hold on; 
        subplot(1,3,2); plot(diff(normW(m,:))); hold on;
        subplot(1,3,3); plot(c_frac(m,:)/(Cxsz(m)^2)); hold on;
    end
    subplot(1,3,1); xlabel('time (x10^{-6})'); ylabel('D_{wee}');
    subplot(1,3,2); xlabel('time (x10^{-6})'); ylabel('dD_{wee}');
    subplot(1,3,3); plot(sum(c_frac,1)/(Cxoffs(end)^2), 'k', 'LineWidth', 2); hold on; 
                    xlabel('time (x10^{-6})'); ylabel('connection fraction per aera (total in black)');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Plot SORN E2I weight matrices 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    idx_saved = 0:10^5:10^6;
    idx_saved(1)=2;
    figure('name', 'SORN E2I Connectivity Matrix');
    for i=1%:10
        filename = sprintf('E2I_Weights_%i.dat',idx_saved(i));
        x = load ([filepath filename]);
        x = spconvert(x);
        %subplot(2,5,i);
        p=imagesc(x);
        xlabel('Excitatory units');
        ylabel('Inhibitory units');
        title(sprintf('t=%ims',idx_saved(i)));
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Plot SORN I2E weight matrices 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    idx_saved = 0:10^6:10^7;
    idx_saved(1)=1;
    figure('Position', [400, 400, 1200, 600], 'name', 'SORN I2E Connectivity Matrix');
    for i=1:length(idx_saved)-1
        filename = sprintf('I2E_Weights_%i.dat',idx_saved(i));
        x = load ([filepath filename]);
        x = spconvert(x);
        subplot(2,5,i);
        p=imagesc(x);
        xlabel('Inhibitory units');
        ylabel('Excitatory units');
        title(sprintf('t=%ims',idx_saved(i)));
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Plot INFOMAX input weigth matrices 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    idx_saved = 0:10^6:6*10^6;%10^7; 
    idx_saved(1)=1;
    normW = zeros(length(Cxsz), length(idx_saved)-1);
    c_fig = figure('Position', [400, 400, 1200, 600], 'name', 'Infomax TH Connectivity Matrix'); % connectivity figure
    wd_fig = figure('Position', [400, 400, 1200, 600], 'name', 'Infomax TH Weights Distribution'); % weight distribution figure
    we_fig  = figure('Position', [400, 400, 1200, 600], 'name', 'Infomax TH Weights Evolution'); % weight evolution figure
    for i=1:length(idx_saved)-1
        filename = sprintf('IMAX_TH_Weights_%i.dat',idx_saved(i));
        x = load ([filepath filename]);
        x = spconvert(x);
        
        % Connectivity
        figure(c_fig);
        subplot(2,5,i);
        p=imagesc(x);
        xlabel(sprintf('t=%ims',idx_saved(i)));

        % Weight distribution
        figure(wd_fig);
        for j=1:4
            subplot(4,10,10*(j-1)+i);
            histogram(x(Cxoffs(j)+1:Cxoffs(j+1),Cxoffs(j)+1:Cxoffs(j+1)));
            xlim([-1 1]);
            ylim([0 1000]);
        end
        xlabel(sprintf('t=%ims', idx_saved(i)));
        
        % Weight evolution
        figure(we_fig);
        for m=1:length(Cxsz)
            normW(m,i) = norm(x(Cxoffs(m)+1:Cxoffs(m+1), Cxoffs(m)+1:Cxoffs(m+1)), 'fro');
        end
        %figure; imagesc(x(Cxoffs(m)+1:Cxoffs(m+1), Cxoffs(m)+1:Cxoffs(m+1)));
    end
    figure(we_fig);
    for m = 1:length(Cxsz)
        subplot(1,2,1); plot(normW(m,:)); hold on; 
        subplot(1,2,2); plot(diff(normW(m,:))); hold on;
    end
    subplot(1,2,1); xlabel('time'); ylabel('D_{w\_infomax}');
    subplot(1,2,2); xlabel('time'); ylabel('dD_{w\_infomax}');
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Plot INFOMAX Lateral weigth matrices 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %filepath='/home/naze//nts/graphs/IBEx/traces_mar20/';
    idx_saved = 0:10^6:10^7;
    idx_saved(1)=1; %idx_saved(2)=50000;
    normW = zeros(length(Cxsz), length(idx_saved)-1);
    c_fig = figure('Position', [400, 400, 1200, 600], 'name', 'Infomax LN Connectivity Matrix'); % connectivity figure
    wd_fig = figure('Position', [400, 400, 1200, 600], 'name', 'Infomax LN Weights Distribution'); % weight distribution figure
    we_fig  = figure('Position', [400, 400, 1200, 600], 'name', 'Infomax LN Weights Evolution'); % weight evolution figure
    for i=1:length(idx_saved)-1
        filename = sprintf('IMAX_LN_Weights_%i.dat',idx_saved(i));
        x = load ([filepath filename]);
        x = spconvert(x);
        %x=eval(filename);
        figure(c_fig);
        subplot(2,5,i);
        %p=imagescT(x);
        p=imagesc(x);
        %caxis([-1, 1]);
        xlabel(sprintf('Infomax LN weights matrix at time step %i',idx_saved(i)));    

        % Weight distribution
        figure(wd_fig);
        for j=1:4
            subplot(4,10,10*(j-1)+i);
            histogram(x(Cxoffs(j)+1:Cxoffs(j+1),Cxoffs(j)+1:Cxoffs(j+1)));
            %xlim([-1 1]);
            %ylim([0 1000]);
        end
        xlabel(sprintf('t=%ims', idx_saved(i)));
        
        % Weight evolution
        figure(we_fig);
        for m=1:length(Cxsz)
            normW(m,i) = norm(x(Cxoffs(m)+1:Cxoffs(m+1), Cxoffs(m)+1:Cxoffs(m+1)), 'fro');
        end
        %figure; imagesc(x(Cxoffs(m)+1:Cxoffs(m+1), Cxoffs(m)+1:Cxoffs(m+1)));
    end
    figure(we_fig);
    for m = 1:length(Cxsz)
        subplot(1,2,1); plot(normW(m,:)); hold on; 
        subplot(1,2,2); plot(diff(normW(m,:))); hold on;
    end
    subplot(1,2,1); xlabel('time'); ylabel('D_{w\_LN} infomax');
    subplot(1,2,2); xlabel('time'); ylabel('dD_{w\_LN} infomax');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Plot Striatum weigths matrices 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    idx_saved(1)=1;
    figure();
    for i=1:1
        filename = sprintf('MSN_LN_Weights_%i.dat',idx_saved(i));
        x = load ([filepath filename]);
        x = spconvert(x);
        p=imagesc(x);
        xlabel('Striatal units');
        ylabel('Striatal units');
        title(sprintf('Striatum Lateral Network connectivity matrix'));
    end


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Plot Cortico-striatal weigths matrices 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    idx_saved = 0:10^5:10^6;
    idx_saved(1)=1;
    c_fig = figure('Position', [400, 400, 1200, 600], 'name', 'Cortico-Striatal Connectivity Matrix'); % connectivity figure
    wd_fig = figure('Position', [400, 400, 1200, 600], 'name', 'Cortico-Striatal Weight Distribution'); % weight distribution figure
    for i=1:length(idx_saved)-1
        filename = sprintf('Cx2Str_Weights_%i.dat',idx_saved(i));
        x = load ([filepath filename]);
        x = spconvert(x);
        figure(c_fig);
        subplot(2,5,i);
        p=imagesc(x);
        caxis([0, 0.08]);
        xlabel(sprintf('time=%ims',idx_saved(i)));

        figure(wd_fig);
        for j=1:4
            subplot(4,10,10*(j-1)+i);
            histogram(x(:,(j-1)*100+1:j*100), 'Normalization', 'pdf', 'BinWidth', 0.0015);
            xlim([0.001 0.03]);
            ylim([0 20]);
        end
        xlabel(sprintf('t=%ims', idx_saved(i)));
    end


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Plot Striato-nigral connectivity matrix 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    idx_saved = 0:10^5:10^6;
    idx_saved(1)=1;
    figure();
    for i=1:1
        filename = sprintf('SN_Weights_%i.dat',idx_saved(i));
        x = load ([filepath filename]);
        x = spconvert(x);
        p=imagesc(x);
        xlabel('Striatal units');
        ylabel('Dopamine units (SNc)');
        title(sprintf('Striato-nigral connectivity matrix'));
    end
end








