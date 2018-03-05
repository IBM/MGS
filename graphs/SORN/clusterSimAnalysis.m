%% Script for SORN simulations analysis after running on cluster

% Flags
plot_figs = 0;
save_figs = 1;
full_analysis = 1;
meta_analysis = 0;

p1 = 'g';
p2 = 'muHIPi';
p3 = 'E2X';
p4 = 'I2X';

% % list of parameters
% param1 = [0.5, 1.0, 1.5];   % etaIP
% param2 = [0.01, 0.02 0.04]; % etaSTDP
% param3 = [0.05, 0.1, 0.2];  % muIP
% param4 = [0.05, 0.1, 0.2];  % ratioHIP
% 
% % List of directory of parameters
% dirs1 = {'g0_5', 'g1_0', 'g1_5'};
% dirs2 = {'muHIPi0_01', 'muHIPi0_02', 'muHIPi0_04'};
% dirs3 = {'E2X0_05', 'E2X0_1', 'E2X0_2'};
% dirs4 = {'I2X0_05', 'I2X0_1', 'I2X0_2'};

% list of parameters
param1 = [1.5];   % g
param2 = [0.01, 0.02, 0.04]; % muHIPi
param3 = [0.05,0.1, 0.2];  % E2X
param4 = [0.05, 0.1, 0.2];  % I2X

% List of directory of parameters
dirs1 = {'g1_5'};
dirs2 = {'muHIPi0_01', 'muHIPi0_02', 'muHIPi0_04'};
dirs3 = {'E2X0_05', 'E2X0_1', 'E2X0_2'};
 dirs4 = {'I2X0_05', 'I2X0_1', 'I2X0_2'};

root_dir = '/home/naze/MGS/graphs/SORN/2018mar03/muIP0_001/ratioIP1_0/etaSTDP0_001/muHIP0_02/ratioHIP0_5';

freqs_log = exp(log(0.1):0.2:log(500));

if full_analysis
    for dir1 = dirs1
        for dir2 = dirs2
            for dir3 = dirs3
                for dir4 = dirs4
                    filepath = fullfile(root_dir, dir1, dir2, dir3, dir4, '0/');
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
    keySet = {'corrCoeff', 'nbClusters', 'Spectrum_linear', 'Spectrum_log', 'mean_degree'};
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
                    filepath = fullfile(root_dir, dir1, dir2, dir3, dir4, '0/');
                    load(strcat(filepath, 'postprocessed_features.mat'));

                    % corr coeff
                    ccoeff = postprocess{1}{2};
                    ccoeff(isnan(ccoeff)) = 0;
                    meanCorrCoeff(i,j,k,l) = mean(mean(full(ccoeff)));

                    % entropy
                    %entropy(i,j,k,l) = postprocess{5}{2};
                    
                    % mean degree
                    degrees = postprocess{5}{2};
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
    

    %% Plot global results
    if plot_figs
        corr_fig = figure('Position', [500 500 800 400], 'Name', 'Correlation');
        degree_fig = figure('Position', [500 500 800 400], 'Name', 'Degree');
        spect_fig = figure('Position', [500 500 800 400], 'Name', 'Spectral Peak'); 
        entropy_fig = figure('Position', [500 500 800 400], 'Name', 'Entropy'); 
        for ll = 1:l
            for jj = 1:j
                % correlations
                figure(corr_fig);
                plt_idx = (l*j)-(j*ll)+jj;
                subtightplot(l,j,plt_idx, [0.01 0.01], [0.15 0.01], [0.125 0.01]);
                imagesc(permute(meanCorrCoeff(:,jj,:,ll), [1 3 2 4]));
                set(gca, 'YDir', 'normal');
                if ll==1 
                    xlabel({p3; strcat(p2,'=',num2str(param2(jj)))}); 
                    set(gca, 'XTick', 1:numel(param3));
                    set(gca, 'XTickLabel', param3);
                else set(gca, 'XTickLabel', []); end
                if jj==1
                    ylabel({strcat(p4,'=', num2str(param4(ll))); 'etaIP'});
                    set(gca, 'YTick', 1:numel(param1));
                    set(gca, 'YTickLabel', param1);
                else set(gca, 'YTickLabel', []); end
                caxis([corr_min corr_max]);
                colormap(flipud(bone));

                % degree
                figure(degree_fig);
                subtightplot(l,j,plt_idx, [0.01 0.01], [0.15 0.01], [0.125 0.01]);
                imagesc(permute(meanDegree(:,jj,:,ll), [1 3 2 4]));
                set(gca, 'YDir', 'normal');
                if ll==1 
                    xlabel({p3; strcat(p2,'=',num2str(param2(jj)))}); 
                    set(gca, 'XTick', 1:numel(param3));
                    set(gca, 'XTickLabel', param3);
                else set(gca, 'XTickLabel', []); end
                if jj==1
                    ylabel({strcat(p4,'=', num2str(param4(ll))); 'etaIP'});
                    set(gca, 'YTick', 1:numel(param1));
                    set(gca, 'YTickLabel', param1);
                else set(gca, 'YTickLabel', []); end
                %caxis([degree_min degree_max]);
                caxis([0 0.4]);
                colormap(flipud(jet));

                % spectrum
                figure(spect_fig);
                subtightplot(l,j,plt_idx, [0.01 0.01], [0.15 0.01], [0.125 0.01]);
                imagesc(permute(peakSpect(:,jj,:,ll), [1 3 2 4]));
                set(gca, 'YDir', 'normal');
                if ll==1 
                    xlabel({p3; strcat(p2,'=',num2str(param2(jj)))}); 
                    set(gca, 'XTick', 1:numel(param3));
                    set(gca, 'XTickLabel', param3);
                else set(gca, 'XTickLabel', []); end
                if jj==1
                    ylabel({strcat(p4,'=', num2str(param4(ll))); 'etaIP'});
                    set(gca, 'YTick', 1:numel(param1));
                    set(gca, 'YTickLabel', param1);
                else set(gca, 'YTickLabel', []); end
                caxis([spect_min spect_max]);
                colormap(parula);
                
%                 % entropy
%                 figure(entropy_fig);
%                 subtightplot(l,j,plt_idx, [0.01 0.01], [0.15 0.01], [0.125 0.01]);
%                 imagesc(permute(entropy(:,jj,:,ll), [1 3 2 4]));
%                 set(gca, 'YDir', 'normal');
%                 if ll==1 
%                     xlabel({'muIP'; strcat('etaSTDP=',num2str(etaSTDP(jj)))}); 
%                     set(gca, 'XTick', 1:numel(muIP));
%                     set(gca, 'XTickLabel', muIP);
%                 else set(gca, 'XTickLabel', []); end
%                 if jj==1
%                     ylabel({strcat('ratioHIP=', num2str(ratioHIP(ll))); 'etaIP'});
%                     set(gca, 'YTick', 1:numel(etaIP));
%                     set(gca, 'YTickLabel', etaIP);
%                 else set(gca, 'YTickLabel', []); end
%                 caxis([entropy_min entropy_max]);
%                 colormap(jet);
            end
        end
    
        if save_figs
            pause(2);
            saveas(corr_fig, fullfile(root_dir, 'corrCoeffs_map'), 'svg');
            saveas(corr_fig, fullfile(root_dir, 'corrCoeffs_map'), 'jpg');
            saveas(degree_fig, fullfile(root_dir, 'degrees_map'), 'svg');
            saveas(degree_fig, fullfile(root_dir, 'degrees_map'), 'jpg');
            saveas(spect_fig, fullfile(root_dir, 'peakSpect_map'), 'svg');
            saveas(spect_fig, fullfile(root_dir, 'peakSpect_map'), 'jpg');
            saveas(entropy_fig, fullfile(root_dir, 'entropy_map'), 'svg');
            saveas(entropy_fig, fullfile(root_dir, 'entropy_map'), 'jpg');
        end
    end
end       
        
        
        
        
        
        
        
        
        
        
        