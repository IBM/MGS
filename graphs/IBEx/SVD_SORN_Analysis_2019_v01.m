%% SVD Analysis on SORN for MGS
Cxsz = [55; 91; 92; 162];
Cxoffs = [0; 55; 146; 238; 400]; 
plot_scatter=1; 
save_figs=1;

t_start=100; t_stop=110; % beginning and end chosen for plotting scatter plots (in sec, from total simulation)
dt=0.001;
% times for SVD analysis (in sec, from begining of simulation), use 10s period by default, for other period, need change in plotting below.. 
t1=100; t2=t1+10; t3=180; t4=t3+10;

%% Scatter plot of L5 spiking activity
figure();
totalUnits = 0 ;
files = {['M1i.txt'], ['Msupi.txt'], ['S1i.txt'], ['Sseci.txt']};
filepath = '/media/nvme/MGS/graphs/IBEx/';  %% your path here if needed
var = {['M1'], ['Msup'], ['S1'], ['Ssec']};
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
    totalUnits = totalUnits + dimX*dimY;
    expr = strcat(var(idx), '_times = data(3:2:end);');
    eval(expr{1});
    expr = strcat(var(idx),'_units = data(4:2:end);');
    eval(expr{1});
    clear data;
    expr = strcat(var(idx),'_idx_start = find(',var(idx),'_times>=(t_start/dt));');
    eval(expr{1});
    expr = strcat(var(idx),'_idx_stop = find(', var(idx), '_times<=(t_stop/dt));');
    eval(expr{1});
    
    offset = Cxoffs(idx)+1;
    expr = strcat('scatter(', var(idx),'_times(',var(idx),'_idx_start(1):',var(idx),'_idx_stop(end))*dt, ',var(idx),'_units(',var(idx),'_idx_start(1):',var(idx),'_idx_stop(end))+offset);');
    eval(expr{1});
    hold on;
end

xlim([t_start, t_stop]);
if save_figs
    pause(1);
    saveas(gcf, strcat(filepath, 'SORN_rasters_', num2str(t1), '_vs_', num2str(t3), 's'), 'svg');
end

xlim([t_start+5.6, t_start+6]);
if save_figs
    pause(1);
    saveas(gcf, strcat(filepath, 'SORN_rasters_', num2str(t1), '_vs_', num2str(t3), 's_zoom'), 'svg');
end


% store results in cells
times = {M1_times, Msup_times, S1_times, Ssec_times};
units = {M1_units, Msup_units, S1_units, Ssec_units};
clear M1_times Msup_times Ssec_times S1_times M1_units Msup_units Ssec_units S1_units;


%% Lower dimensional manifold using SVD
t_artifact=0.5;
tauZ=100; 
t_beg = {t1-t_artifact, t3-t_artifact}; t_end={t2+t_artifact, t4+t_artifact};
boxcar=ones(tauZ,1); % sliding window for box car filtering

% create cell to store full matrices of data (in ms, however times in sec so need to *1000, and add tauZ for "boxcar" filtering)
epoch1 = zeros(round((t_end{1}-t_beg{1})/dt+tauZ+1), totalUnits);
epoch2 = zeros(round((t_end{2}-t_beg{2})/dt+tauZ+1), totalUnits);
epochs = {epoch1, epoch2};
clear epoch1 epoch2;

gauss_conv = normpdf(-round(tauZ/2):round(tauZ/2), 0, 20);  % gaussian kernel to convolute signal (based on tauZ=100)

% transforms data (sparse matrices) into full matrix
for ep=1:length(epochs)
    for m=1:length(var)
        t_idx = find(times{m}>=t_beg{ep}/dt & times{m}<=t_end{ep}/dt);
        for i=1:Cxsz(m)
            unit_i_idx = find(units{m}(t_idx)==(i-1));
            t_i_idx = t_idx(unit_i_idx);
            t_i_idx_full = round(times{m}(t_i_idx)-t_beg{ep}/dt);
            epochs{ep}(t_i_idx_full+1,Cxoffs(m)+i)=1; % +1 for matlab (idx 0 exists in MGS)
        end
    end
    % filter epochs using gaussian kernel convolution (better than boxcar filtering)
    for ts=1:(size(epochs{ep},1)-tauZ)
        epochs{ep}(ts,:) = gauss_conv * epochs{ep}(ts:ts+tauZ,:);
    end
    %epochs{ep} = conv2(epochs{ep}, boxcar, 'same');
    epochs{ep} = epochs{ep}(t_artifact/dt+tauZ:end-t_artifact/dt, :); % remove artifact of convolution
end


comp=[1,2,3];  % which singular values to plot 
nAreas=length(var);
eigenval = cell(2,nAreas); % row1: t1 to t2; row2: t3 to t4


figure('Units', 'normalized', 'Position', [0 0 0.6 0.6]);
% Set Info Text Box
textBox = uicontrol('style', 'text');
figname=sprintf('SVD_L5_SORN_comp%ivs%ivs%i', comp(1), comp(2), comp(3));
set(textBox, 'String', figname, 'FontSize', 14);
set(textBox, 'Units', 'normalized', 'Position', [0 0.95 1 0.05]); % position [left bottow width height]

first_USV=cell(3,nAreas);
second_USV=cell(3,nAreas);

for m=1:nAreas
    [first_USV{1,m}, first_USV{2,m}, first_USV{3,m}] = svd(epochs{1}(:,Cxoffs(m)+1:Cxoffs(m+1)));
    [second_USV{1,m}, second_USV{2,m}, second_USV{3,m}] = svd(epochs{2}(:,Cxoffs(m)+1:Cxoffs(m+1)));
    
    % histograms of singular values distributions
    subplot('Position', [(m-1)/4 0.66 0.125 0.33]);
    eigenval{1,m}=diag(first_USV{2,m});
    eigenval{1,m}=eigenval{1,m}/sum(eigenval{1,m}); %% show contribution of singular values in %
    bar(eigenval{1,m}(comp(1):end));
        
    subplot('Position', [(m-1)/4+0.125 0.66 0.125 0.33]);    
    eigenval{2,m}=diag(second_USV{2,m});
    eigenval{2,m}=eigenval{2,m}/sum(eigenval{2,m}); %% show contribution of singular values in %
    bar(eigenval{2,m}(comp(1):end));
    
    % Phase space trajectories (colored for a 10s period)
    subplot('Position', [(m-1)/4 0.33 0.25 0.33]);    
    plot3(first_USV{1,m}(1:2000,comp(1)),first_USV{1,m}(1:2000,comp(2)),first_USV{1,m}(1:2000,comp(3)),'.b');
    hold on;
    plot3(first_USV{1,m}(2001:4000,comp(1)),first_USV{1,m}(2001:4000,comp(2)),first_USV{1,m}(2001:4000,comp(3)),'.g');
    hold on;
    plot3(first_USV{1,m}(4001:6000,comp(1)),first_USV{1,m}(4001:6000,comp(2)),first_USV{1,m}(4001:6000,comp(3)),'.r');
    hold on;
    plot3(first_USV{1,m}(6001:8000,comp(1)),first_USV{1,m}(6001:8000,comp(2)),first_USV{1,m}(6001:8000,comp(3)),'.m');
    hold on;
    plot3(first_USV{1,m}(8001:10000,comp(1)),first_USV{1,m}(8001:10000,comp(2)),first_USV{1,m}(8001:10000,comp(3)),'.k');
        
    subplot('Position', [(m-1)/4 0 0.25 0.33]);    
    plot3(second_USV{1,m}(1:2000,comp(1)),second_USV{1,m}(1:2000,comp(2)),second_USV{1,m}(1:2000,comp(3)),'.b');
    hold on;
    plot3(second_USV{1,m}(2001:4000,comp(1)),second_USV{1,m}(2001:4000,comp(2)),second_USV{1,m}(2001:4000,comp(3)),'.g');
    hold on;
    plot3(second_USV{1,m}(4001:6000,comp(1)),second_USV{1,m}(4001:6000,comp(2)),second_USV{1,m}(4001:6000,comp(3)),'.r');
    hold on;
    plot3(second_USV{1,m}(6001:8000,comp(1)),second_USV{1,m}(6001:8000,comp(2)),second_USV{1,m}(6001:8000,comp(3)),'.m');
    hold on;
    plot3(second_USV{1,m}(8001:10000,comp(1)),second_USV{1,m}(8001:10000,comp(2)),second_USV{1,m}(8001:10000,comp(3)),'.k');

end

if save_figs
    pause(1);
    saveas(gcf, strcat(filepath, 'SORN_SVD_3D_', num2str(t1), '_vs_', num2str(t3), 's'), 'svg');
end
