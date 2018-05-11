%% Script for SORN simulations analysis after running on cluster

% Flags
full_analysis = 0;
meta_analysis = 1;
plot_figs = 1;
save_figs = 1;
save_array_maps = 1;

p1 = 'g';
p2 = 'tauSTDP';
p3 = 'muDelay';
p4 = 'ratioDelay';

% List of parameters
param1 = [1.0];  
param2 = [0.9, 0.99]; 
param3 = [1 ,2, 3, 5]; 
param4 = [0.5 1];  % ratioHIP
 
% List of directory of parameters
dirs1 = {'g1_0'};
dirs2 = {'tauSTDP0_9', 'tauSTDP0_99'};
dirs3 = {'muDelay1', 'muDelay2', 'muDelay3', 'muDelay5'};
dirs4 = {'ratioDelay0_5', 'ratioDelay1_0'};


root_dir = '/home/naze/MGS/graphs/SORN/2018mar24';
base1dir = 'muIP0_0001/ratioIP1_0/etaSTDP0_001';
base2dir = 'muHIP0_01/ratioHIP0_5/muHIPi0_02/ratioHIPi0_5';
base3dir = 'E2X0_05/I2X0_1/';

freqs_log = exp(log(0.1):0.2:log(500));

if full_analysis
    for dir1 = dirs1
        for dir2 = dirs2
            for dir3 = dirs3
                for dir4 = dirs4
                    filepath = fullfile(root_dir, base1dir, dir2, base2dir, dir1, base3dir, dir3, dir4, '0/');
                    DataProcessSORN;
                end
            end
        end
    end
end


%% Get global results
if meta_analysis
    n_sim = numel(dirs1) + numel(dirs2) + numel(dirs3) + numel(dirs4);
    % set labels using containers (need better saving to be automated...)
    keySet = {'corrCoeff', 'nbClusters', 'Spectrum_linear', 'Spectrum_log', 'entropy', 'mean_degree'};
    meanCorrCoeff = zeros(numel(dirs1), numel(dirs2), numel(dirs3), numel(dirs4));
    meanDegree = zeros(numel(dirs1), numel(dirs2), numel(dirs3), numel(dirs4));
    peakSpect = zeros(numel(dirs1), numel(dirs2), numel(dirs3), numel(dirs4));
    entropy = zeros(numel(dirs1), numel(dirs2), numel(dirs3), numel(dirs4));
    % loading the results from simulation and extract content
    for i = 1:numel(dirs1)
        dir1 = dirs1{i};
        for j = 1:numel(dirs2)
            dir2 = dirs2{j};
            for k = 1:numel(dirs3)
                dir3 = dirs3{k};
                for l = 1:numel(dirs4)
                    dir4 = dirs4{l};
                    clear postprocess;
                    filepath = fullfile(root_dir, base1dir, dir2, base2dir, dir1, base3dir, dir3, dir4, '0/');
                    load(strcat(filepath, 'postprocessed_features.mat'));

                    % corr coeff
                    ccoeff = postprocess{1}{2};
                    ccoeff(isnan(ccoeff)) = 0;
                    meanCorrCoeff(i,j,k,l) = mean(mean(full(ccoeff)));

                    % entropy
                    entropy(i,j,k,l) = postprocess{5}{2};
                    
                    % mean degree
                    degrees = postprocess{6}{2};
                    meanDegree(i,j,k,l) = mean(degrees);

                    % peak spectrum
                    Pxx_log = postprocess{4}{2};
                   % freqs_log = postprocess{4}{3};
                    [~, max_idx] = max(Pxx_log{1}); %Pxx_log{1} : exc cells
                    peakSpect(i,j,k,l) = freqs_log(max_idx);

                end
            end
        end
    end
    corr_min = min(min(min(min(meanCorrCoeff))));
    corr_max = max(max(max(max(meanCorrCoeff))));
    degree_min = min(min(min(min(meanDegree))));
    degree_max = max(max(max(max(meanDegree))));
    spect_min = min(min(min(min(peakSpect))));
    spect_max = max(max(max(max(peakSpect))));
    entropy_min = min(min(min(min(entropy))));
    entropy_max = max(max(max(max(entropy))));
    
    if save_array_maps
        entropy(isnan(entropy)) = 0;
        save(fullfile(root_dir, 'entropy_array_map.mat'), 'entropy', '-mat'); 
        
    end
    

    %% Plot global results
    if plot_figs
        corr_fig = figure('Position', [500 500 800 400], 'Name', 'Correlation');
        degree_fig = figure('Position', [500 500 800 400], 'Name', 'Degree');
        spect_fig = figure('Position', [500 500 800 400], 'Name', 'Spectral Peak'); 
        entropy_fig = figure('Position', [500 500 800 400], 'Name', 'Entropy'); 
        for out_row = 1:i
            for out_col = 1:j
                % correlations
                figure(corr_fig);
                plt_idx = (i*j)-(j*out_row)+out_col;
                subtightplot(i,j,plt_idx, [0.01 0.01], [0.15 0.01], [0.125 0.01]);
                imagesc(permute(meanCorrCoeff(out_row,out_col,:,:), [3 4 1 2]));
                set(gca, 'YDir', 'normal');
                if out_row==1 
                    xlabel({p4; strcat(p2,'=',num2str(param2(out_col)))}); 
                    set(gca, 'XTick', 1:numel(param4));
                    set(gca, 'XTickLabel', param4);
                else set(gca, 'XTickLabel', []); end
                if out_col==1
                    ylabel({strcat(p1,'=', num2str(param1(out_row))); p3});
                    set(gca, 'YTick', 1:numel(param3));
                    set(gca, 'YTickLabel', param3);
                else set(gca, 'YTickLabel', []); end
                caxis([corr_min corr_max]);
                colormap(flipud(bone));

                % degree
                figure(degree_fig);
                subtightplot(i,j,plt_idx, [0.01 0.01], [0.15 0.01], [0.125 0.01]);
                imagesc(permute(meanDegree(out_row,out_col,:,:), [3 4 1 2]));
                set(gca, 'YDir', 'normal');
                if out_row==1 
                    xlabel({p4; strcat(p2,'=',num2str(param2(out_col)))}); 
                    set(gca, 'XTick', 1:numel(param2));
                    set(gca, 'XTickLabel', param2);
                else set(gca, 'XTickLabel', []); end
                if out_col==1
                    ylabel({strcat(p1,'=', num2str(param1(out_row))); p3});
                    set(gca, 'YTick', 1:numel(param3));
                    set(gca, 'YTickLabel', param3);
                else set(gca, 'YTickLabel', []); end
                caxis([degree_min degree_max]);
                %caxis([0 0.2]);
                colormap(winter);

                % spectrum
                figure(spect_fig);
                subtightplot(i,j,plt_idx, [0.01 0.01], [0.15 0.01], [0.125 0.01]);
                imagesc(permute(peakSpect(out_row,out_col,:,:), [3 4 1 2]));
                set(gca, 'YDir', 'normal');
                if out_row==1 
                    xlabel({p4; strcat(p2,'=',num2str(param2(out_col)))}); 
                    set(gca, 'XTick', 1:numel(param4));
                    set(gca, 'XTickLabel', param4);
                else set(gca, 'XTickLabel', []); end
                if out_col==1
                    ylabel({strcat(p1,'=', num2str(param1(out_row))); p3});
                    set(gca, 'YTick', 1:numel(param3));
                    set(gca, 'YTickLabel', param3);
                else set(gca, 'YTickLabel', []); end
                caxis([spect_min spect_max]);
                colormap(parula);
                
                % entropy
                figure(entropy_fig);
                subtightplot(i,j,plt_idx, [0.01 0.01], [0.15 0.01], [0.125 0.01]);
                imagesc(permute(entropy(out_row,out_col,:,:), [3 4 1 2]));
                set(gca, 'YDir', 'normal');
                if out_row==1 
                    xlabel({p4; strcat(p2, '=',num2str(param2(out_col)))}); 
                    set(gca, 'XTick', 1:numel(param4));
                    set(gca, 'XTickLabel', param4);
                else set(gca, 'XTickLabel', []); end
                if out_col==1
                    ylabel({strcat(p1,'=', num2str(param1(out_row))); p3});
                    set(gca, 'YTick', 1:numel(param3));
                    set(gca, 'YTickLabel', param3);
                else set(gca, 'YTickLabel', []); end
                caxis([entropy_min entropy_max]);
                colormap(jet);
            end
        end
    
        if save_figs
            pause(2);
            saveas(corr_fig, fullfile(root_dir, 'corrCoeffs_map'), 'svg');
            saveas(corr_fig, fullfile(root_dir, 'corrCoeffs_map'), 'jpg');
            print(corr_fig, fullfile(root_dir, 'corrCoeffs_map'), '-dpng');
            
            saveas(degree_fig, fullfile(root_dir, 'degrees_map'), 'svg');
            saveas(degree_fig, fullfile(root_dir, 'degrees_map'), 'jpg');
            print(degree_fig, fullfile(root_dir, 'degree_map'), '-dpng');
            
            saveas(spect_fig, fullfile(root_dir, 'peakSpect_map'), 'svg');
            saveas(spect_fig, fullfile(root_dir, 'peakSpect_map'), 'jpg');
            print(spect_fig, fullfile(root_dir, 'peakSpect_map'), '-dpng');
            
            saveas(entropy_fig, fullfile(root_dir, 'entropy_map'), 'svg');
            saveas(entropy_fig, fullfile(root_dir, 'entropy_map'), 'jpg');
            print(entropy_fig, fullfile(root_dir, 'entropy_map'), '-dpng');
        end
    end
end       
        
        
        
        
        
        
        
        
        
        
        