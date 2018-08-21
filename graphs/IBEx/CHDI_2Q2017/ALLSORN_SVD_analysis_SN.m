%% SVD analysis of SORN (for x_s and u_s, see x_analysis_4_areas.m)
%  HOW TO USE: 
%       1) prerequisite: simulation data must be loaded
%       2) set t1 and t3 to the beginning of the intervals to be
%   analyzed (in ms), by default SVD is performed on 10000 tsteps
%       3) run
%
%  Outputs:
%       - One figure for SV histograms and 3D SVD projections phases spaces (fist: top, second: bottom)
%       - Two figures for 2D SVD phase spaces (one for each interval).
%       4x3 plots within figure => 3 projections (SV1-SV2, SV1-SV3, SV2-SV3) for each of the 4 SORN areas
 
%SVD_areas = nAreas;
SVD_areas = 4;
interval_length = 10000; 
% first time interval to analyze 
t1=620000; t2=t1+interval_length;
first_USV=cell(3,SVD_areas);

% second time interval to analyze
t3=630000; t4=t3+interval_length;
second_USV=cell(3,SVD_areas);

% flags for color coded trajectories when tone present
oneToneOn = 0;  
twoTonesOn = 0;
thirty_kts = 0;
one_kts = 0;
nine_kts = 1;
six_kts = 0;

% flags for plotting options
traj_2D = 1;
entropy_bar= 0;
wvlt_spect = 0;
fit_distr = 0;

comp=[1,2,3]; % which components of the SVD to plot against each other
eigenval = cell(2,SVD_areas); % rows: first & second SVD

% extract and filter epochs
gauss_width = 100;
epoch1=SFRE_raster(t1-gauss_width:t2,:);
epoch2=SFRE_raster(t3-gauss_width:t4,:);
gauss_conv = normpdf(-gauss_width/2:gauss_width/2, 0, 20);  % gaussian kernel to convolute signal (tauX=100)
for i=1:(size(epoch1,1)-gauss_width)    
    epoch1(i,:) = gauss_conv * epoch1(i:i+gauss_width,:);
    epoch2(i,:) = gauss_conv * epoch2(i:i+gauss_width,:);
end
epoch1=epoch1(1:end-gauss_width,:);
epoch2=epoch2(1:end-gauss_width,:);


figure('Units', 'normalized', 'Position', [0 0 0.5 0.5]);
textBox = uicontrol('style', 'text');
figname=sprintf('SVD_SORN_comp%ivs%ivs%i_Ach_%3i | June06', comp(1), comp(2), comp(3), Ach_2*100);
set(textBox, 'String', figname, 'FontSize', 14);
set(textBox, 'Units', 'normalized', 'Position', [0 0.95 1 0.05]); % position [left bottow width height]


for m=1:SVD_areas
    if SVD_areas > 1
        [first_USV{1,m}, first_USV{2,m}, first_USV{3,m}] = svd(epoch1(:,Cxoffs(m)+1:Cxoffs(m+1)));
        [second_USV{1,m}, second_USV{2,m}, second_USV{3,m}] = svd(epoch2(:,Cxoffs(m)+1:Cxoffs(m+1)));
    else
        [first_USV{1,m}, first_USV{2,m}, first_USV{3,m}] = svd(epoch1);
        [second_USV{1,m}, second_USV{2,m}, second_USV{3,m}] = svd(epoch2);
    end
    
    % histograms of SV contributions
    subplot('Position', [(m-1)/SVD_areas 0.66 1/(2*SVD_areas) 0.33]);
    eigenval{1,m}=diag(first_USV{2,m});
    eigenval{1,m}=eigenval{1,m}/sum(eigenval{1,m}); %% show contribution of singular values in %
    bar(eigenval{1,m}(comp(1):end));
    
    subplot('Position', [(m-1)/SVD_areas+1/(2*SVD_areas) 0.66 1/(2*SVD_areas) 0.33]);    
    eigenval{2,m}=diag(second_USV{2,m});
    eigenval{2,m}=eigenval{2,m}/sum(eigenval{2,m}); %% show contribution of singular values in %
    bar(eigenval{2,m}(comp(1):end));
    
    % colorcoded phase spaces (different according to stimulus modalities i.e. when tone appear, red or blue trajectories are displayed)
    if oneToneOn
        subplot('Position', [(m-1)/SVD_areas 0.33 1/SVD_areas 0.33]);    
        plot3(first_USV{1,m}(1:2500,comp(1)),first_USV{1,m}(1:2500,comp(2)),first_USV{1,m}(1:2500,comp(3)),'.k');
        hold on;
        plot3(first_USV{1,m}(2501:3500,comp(1)),first_USV{1,m}(2501:3500,comp(2)),first_USV{1,m}(2501:3500,comp(3)),'.r');
        hold on;
        plot3(first_USV{1,m}(3501:7500,comp(1)),first_USV{1,m}(3501:7500,comp(2)),first_USV{1,m}(3501:7500,comp(3)),'.k');
        hold on;
        plot3(first_USV{1,m}(7501:8500,comp(1)),first_USV{1,m}(7501:8500,comp(2)),first_USV{1,m}(7501:8500,comp(3)),'.b');
        hold on;
        plot3(first_USV{1,m}(8501:10000,comp(1)),first_USV{1,m}(8501:10000,comp(2)),first_USV{1,m}(8501:10000,comp(3)),'.k');
        
        subplot('Position', [(m-1)/SVD_areas 0 1/SVD_areas 0.33]);    
        plot3(second_USV{1,m}(1:2500,comp(1)),second_USV{1,m}(1:2500,comp(2)),second_USV{1,m}(1:2500,comp(3)),'.k');
        hold on;
        plot3(second_USV{1,m}(2501:3500,comp(1)),second_USV{1,m}(2501:3500,comp(2)),second_USV{1,m}(2501:3500,comp(3)),'.r');
        hold on;
        plot3(second_USV{1,m}(3501:7500,comp(1)),second_USV{1,m}(3501:7500,comp(2)),second_USV{1,m}(3501:7500,comp(3)),'.k');
        hold on;
        plot3(second_USV{1,m}(7501:8500,comp(1)),second_USV{1,m}(7501:8500,comp(2)),second_USV{1,m}(7501:8500,comp(3)),'.b');
        hold on;
        plot3(second_USV{1,m}(8501:10000,comp(1)),second_USV{1,m}(8501:10000,comp(2)),second_USV{1,m}(8501:10000,comp(3)),'.k');
    
    elseif twoTonesOn
        intervals=[[1,1000]; [1001,2000]; [2001,3000]; [3001,4001]; [4001,5000]; [5001,6000]; [6001,7000]; [7001,8000]; [8001,9001]; [9001,10000]];
        subplot('Position', [(m-1)/SVD_areas 0.33 1/SVD_areas 0.33]);    
        plot3(first_USV{1,m}(intervals(1,1):intervals(1,2),comp(1)),first_USV{1,m}(intervals(1,1):intervals(1,2),comp(2)),first_USV{1,m}(intervals(1,1):intervals(1,2),comp(3)),'.k');
        hold on;
        plot3(first_USV{1,m}(intervals(2,1):intervals(2,2),comp(1)),first_USV{1,m}(intervals(2,1):intervals(2,2),comp(2)),first_USV{1,m}(intervals(2,1):intervals(2,2),comp(3)),'.r');
        hold on;
        plot3(first_USV{1,m}(intervals(3,1):intervals(3,2),comp(1)),first_USV{1,m}(intervals(3,1):intervals(3,2),comp(2)),first_USV{1,m}(intervals(3,1):intervals(3,2),comp(3)),'.k');
        hold on;
        plot3(first_USV{1,m}(intervals(4,1):intervals(4,2),comp(1)),first_USV{1,m}(intervals(4,1):intervals(4,2),comp(2)),first_USV{1,m}(intervals(4,1):intervals(4,2),comp(3)),'.b');
        hold on;
        plot3(first_USV{1,m}(intervals(5,1):intervals(5,2),comp(1)),first_USV{1,m}(intervals(5,1):intervals(5,2),comp(2)),first_USV{1,m}(intervals(5,1):intervals(5,2),comp(3)),'.k');
        hold on;
        plot3(first_USV{1,m}(intervals(6,1):intervals(6,2),comp(1)),first_USV{1,m}(intervals(6,1):intervals(6,2),comp(2)),first_USV{1,m}(intervals(6,1):intervals(6,2),comp(3)),'.k');
        hold on;
        plot3(first_USV{1,m}(intervals(7,1):intervals(7,2),comp(1)),first_USV{1,m}(intervals(7,1):intervals(7,2),comp(2)),first_USV{1,m}(intervals(7,1):intervals(7,2),comp(3)),'.r');
        hold on;
        plot3(first_USV{1,m}(intervals(8,1):intervals(8,2),comp(1)),first_USV{1,m}(intervals(8,1):intervals(8,2),comp(2)),first_USV{1,m}(intervals(8,1):intervals(8,2),comp(3)),'.k');
        hold on;
        plot3(first_USV{1,m}(intervals(9,1):intervals(9,2),comp(1)),first_USV{1,m}(intervals(9,1):intervals(9,2),comp(2)),first_USV{1,m}(intervals(9,1):intervals(9,2),comp(3)),'.b');
        hold on;
        plot3(first_USV{1,m}(intervals(10,1):intervals(10,2),comp(1)),first_USV{1,m}(intervals(10,1):intervals(10,2),comp(2)),first_USV{1,m}(intervals(10,1):intervals(10,2),comp(3)),'.k');
        
        subplot('Position', [(m-1)/SVD_areas 0 1/SVD_areas 0.33]);    
        plot3(second_USV{1,m}(intervals(1,1):intervals(1,2),comp(1)),second_USV{1,m}(intervals(1,1):intervals(1,2),comp(2)),second_USV{1,m}(intervals(1,1):intervals(1,2),comp(3)),'.k');
        hold on;
        plot3(second_USV{1,m}(intervals(2,1):intervals(2,2),comp(1)),second_USV{1,m}(intervals(2,1):intervals(2,2),comp(2)),second_USV{1,m}(intervals(2,1):intervals(2,2),comp(3)),'.r');
        hold on;
        plot3(second_USV{1,m}(intervals(3,1):intervals(3,2),comp(1)),second_USV{1,m}(intervals(3,1):intervals(3,2),comp(2)),second_USV{1,m}(intervals(3,1):intervals(3,2),comp(3)),'.k');
        hold on;
        plot3(second_USV{1,m}(intervals(4,1):intervals(4,2),comp(1)),second_USV{1,m}(intervals(4,1):intervals(4,2),comp(2)),second_USV{1,m}(intervals(4,1):intervals(4,2),comp(3)),'.b');
        hold on;
        plot3(second_USV{1,m}(intervals(5,1):intervals(5,2),comp(1)),second_USV{1,m}(intervals(5,1):intervals(5,2),comp(2)),second_USV{1,m}(intervals(5,1):intervals(5,2),comp(3)),'.k');
        hold on;
        plot3(second_USV{1,m}(intervals(6,1):intervals(6,2),comp(1)),second_USV{1,m}(intervals(6,1):intervals(6,2),comp(2)),second_USV{1,m}(intervals(6,1):intervals(6,2),comp(3)),'.k');
        hold on;
        plot3(second_USV{1,m}(intervals(7,1):intervals(7,2),comp(1)),second_USV{1,m}(intervals(7,1):intervals(7,2),comp(2)),second_USV{1,m}(intervals(7,1):intervals(7,2),comp(3)),'.r');
        hold on;
        plot3(second_USV{1,m}(intervals(8,1):intervals(8,2),comp(1)),second_USV{1,m}(intervals(8,1):intervals(8,2),comp(2)),second_USV{1,m}(intervals(8,1):intervals(8,2),comp(3)),'.k');
        hold on;
        plot3(second_USV{1,m}(intervals(9,1):intervals(9,2),comp(1)),second_USV{1,m}(intervals(9,1):intervals(9,2),comp(2)),second_USV{1,m}(intervals(9,1):intervals(9,2),comp(3)),'.b');
        hold on;
        plot3(second_USV{1,m}(intervals(10,1):intervals(10,2),comp(1)),second_USV{1,m}(intervals(10,1):intervals(10,2),comp(2)),second_USV{1,m}(intervals(10,1):intervals(10,2),comp(3)),'.k');
        
    
    elseif thirty_kts
        subplot('Position', [(m-1)/SVD_areas 0.33 1/SVD_areas 0.33]);    
        plot3(first_USV{1,m}(1:10000,comp(1)),first_USV{1,m}(1:10000,comp(2)),first_USV{1,m}(1:10000,comp(3)),'.k');
        hold on;
        plot3(first_USV{1,m}(10001:20000,comp(1)),first_USV{1,m}(10001:20000,comp(2)),first_USV{1,m}(10001:20000,comp(3)),'.r');
        hold on;
        plot3(first_USV{1,m}(20001:30000,comp(1)),first_USV{1,m}(20001:30000,comp(2)),first_USV{1,m}(20001:30000,comp(3)),'.b');
        hold on;
    
        subplot('Position', [(m-1)/SVD_areas 0 1/SVD_areas 0.33]);    
        plot3(second_USV{1,m}(1:10000,comp(1)),second_USV{1,m}(1:10000,comp(2)),second_USV{1,m}(1:10000,comp(3)),'.k');
        hold on;
        plot3(second_USV{1,m}(10001:20000,comp(1)),second_USV{1,m}(10001:20000,comp(2)),second_USV{1,m}(10001:20000,comp(3)),'.r');
        hold on;
        plot3(second_USV{1,m}(20001:30000,comp(1)),second_USV{1,m}(20001:30000,comp(2)),second_USV{1,m}(20001:30000,comp(3)),'.b');
        hold on;
    
    elseif one_kts
        subplot('Position', [(m-1)/SVD_areas 0.33 1/SVD_areas 0.33]);    
        plot3(first_USV{1,m}(1:200,comp(1)),first_USV{1,m}(1:200,comp(2)),first_USV{1,m}(1:200,comp(3)),'.r');
        hold on;
        plot3(first_USV{1,m}(201:400,comp(1)),first_USV{1,m}(201:400,comp(2)),first_USV{1,m}(201:400,comp(3)),'.k');
        hold on;
        plot3(first_USV{1,m}(401:600,comp(1)),first_USV{1,m}(401:600,comp(2)),first_USV{1,m}(401:600,comp(3)),'.b');
        hold on;
        plot3(first_USV{1,m}(601:800,comp(1)),first_USV{1,m}(601:800,comp(2)),first_USV{1,m}(601:800,comp(3)),'.m');
        hold on;
        plot3(first_USV{1,m}(801:1000,comp(1)),first_USV{1,m}(801:1000,comp(2)),first_USV{1,m}(801:1000,comp(3)),'.g');
        
        subplot('Position', [(m-1)/SVD_areas 0 1/SVD_areas 0.33]);    
        plot3(second_USV{1,m}(1:200,comp(1)),second_USV{1,m}(1:200,comp(2)),second_USV{1,m}(1:200,comp(3)),'.r');
        hold on;
        plot3(second_USV{1,m}(201:400,comp(1)),second_USV{1,m}(201:400,comp(2)),second_USV{1,m}(201:400,comp(3)),'.k');
        hold on;
        plot3(second_USV{1,m}(401:600,comp(1)),second_USV{1,m}(401:600,comp(2)),second_USV{1,m}(401:600,comp(3)),'.b');
        hold on;
        plot3(second_USV{1,m}(601:800,comp(1)),second_USV{1,m}(601:800,comp(2)),second_USV{1,m}(601:800,comp(3)),'.m');
        hold on;
        plot3(second_USV{1,m}(801:1000,comp(1)),second_USV{1,m}(801:1000,comp(2)),second_USV{1,m}(801:1000,comp(3)),'.g');
    
    elseif nine_kts
        subplot('Position', [(m-1)/SVD_areas 0.33 1/SVD_areas 0.33]);    
        plot3(first_USV{1,m}(1:3000,comp(1)),first_USV{1,m}(1:3000,comp(2)),first_USV{1,m}(1:3000,comp(3)),'.r');
        hold on;
        plot3(first_USV{1,m}(3001:6000,comp(1)),first_USV{1,m}(3001:6000,comp(2)),first_USV{1,m}(3001:6000,comp(3)),'.b');
        hold on;
        plot3(first_USV{1,m}(6001:9000,comp(1)),first_USV{1,m}(6001:9000,comp(2)),first_USV{1,m}(6001:9000,comp(3)),'.g');
        
        subplot('Position', [(m-1)/SVD_areas 0 1/SVD_areas 0.33]);    
        plot3(second_USV{1,m}(1:3000,comp(1)),second_USV{1,m}(1:3000,comp(2)),second_USV{1,m}(1:3000,comp(3)),'.r');
        hold on;
        plot3(second_USV{1,m}(3001:6000,comp(1)),second_USV{1,m}(3001:6000,comp(2)),second_USV{1,m}(3001:6000,comp(3)),'.b');
        hold on;
        plot3(second_USV{1,m}(6001:9000,comp(1)),second_USV{1,m}(6001:9000,comp(2)),second_USV{1,m}(6001:9000,comp(3)),'.g');
    
    elseif six_kts
        subplot('Position', [(m-1)/SVD_areas 0.33 1/SVD_areas 0.33]);    
        plot3(first_USV{1,m}(1:2000,comp(1)),first_USV{1,m}(1:2000,comp(2)),first_USV{1,m}(1:2000,comp(3)),'.r');
        hold on;
        plot3(first_USV{1,m}(2001:4000,comp(1)),first_USV{1,m}(2001:4000,comp(2)),first_USV{1,m}(2001:4000,comp(3)),'.b');
        hold on;
        plot3(first_USV{1,m}(4001:6000,comp(1)),first_USV{1,m}(4001:6000,comp(2)),first_USV{1,m}(4001:6000,comp(3)),'.g');
        
        subplot('Position', [(m-1)/SVD_areas 0 0.25 0.33]);    
        plot3(second_USV{1,m}(1:2000,comp(1)),second_USV{1,m}(1:2000,comp(2)),second_USV{1,m}(1:2000,comp(3)),'.r');
        hold on;
        plot3(second_USV{1,m}(2001:4000,comp(1)),second_USV{1,m}(2001:4000,comp(2)),second_USV{1,m}(2001:4000,comp(3)),'.b');
        hold on;
        plot3(second_USV{1,m}(4001:6000,comp(1)),second_USV{1,m}(4001:6000,comp(2)),second_USV{1,m}(4001:6000,comp(3)),'.g');
    
    else
        subplot('Position', [(m-1)/SVD_areas 0.33 1/SVD_areas 0.33]);    
        plot3(first_USV{1,m}(1:2000,comp(1)),first_USV{1,m}(1:2000,comp(2)),first_USV{1,m}(1:2000,comp(3)),'.r');
        hold on;
        plot3(first_USV{1,m}(2001:4000,comp(1)),first_USV{1,m}(2001:4000,comp(2)),first_USV{1,m}(2001:4000,comp(3)),'.k');
        hold on;
        plot3(first_USV{1,m}(4001:6000,comp(1)),first_USV{1,m}(4001:6000,comp(2)),first_USV{1,m}(4001:6000,comp(3)),'.b');
        hold on;
        plot3(first_USV{1,m}(6001:8000,comp(1)),first_USV{1,m}(6001:8000,comp(2)),first_USV{1,m}(6001:8000,comp(3)),'.m');
        hold on;
        plot3(first_USV{1,m}(8001:10000,comp(1)),first_USV{1,m}(8001:10000,comp(2)),first_USV{1,m}(8001:10000,comp(3)),'.g');
        
        subplot('Position', [(m-1)/SVD_areas 0 1/SVD_areas 0.33]);    
        plot3(second_USV{1,m}(1:2000,comp(1)),second_USV{1,m}(1:2000,comp(2)),second_USV{1,m}(1:2000,comp(3)),'.r');
        hold on;
        plot3(second_USV{1,m}(2001:4000,comp(1)),second_USV{1,m}(2001:4000,comp(2)),second_USV{1,m}(2001:4000,comp(3)),'.k');
        hold on;
        plot3(second_USV{1,m}(4001:6000,comp(1)),second_USV{1,m}(4001:6000,comp(2)),second_USV{1,m}(4001:6000,comp(3)),'.b');
        hold on;
        plot3(second_USV{1,m}(6001:8000,comp(1)),second_USV{1,m}(6001:8000,comp(2)),second_USV{1,m}(6001:8000,comp(3)),'.m');
        hold on;
        plot3(second_USV{1,m}(8001:10000,comp(1)),second_USV{1,m}(8001:10000,comp(2)),second_USV{1,m}(8001:10000,comp(3)),'.g');
    end
end

%% Colored 2D projections of SVs
if traj_2D
    if oneToneOn 
        figure; 
        idxplot=1;
        for i=1:SVD_areas
            for j = 1:3 % SVD components proj combinations (1-2, 1-3, 2-3)
                for k = j+1:3
                    subplot(SVD_areas,3,idxplot); 
                    plot(first_USV{1,i}(1:2500,j), first_USV{1,i}(1:2500,k), '.k'); hold on;
                    plot(first_USV{1,i}(2501:3500,j), first_USV{1,i}(2501:3500,k), '.r'); hold on;
                    plot(first_USV{1,i}(3501:7500,j), first_USV{1,i}(3501:7500,k), '.k'); hold on;
                    plot(first_USV{1,i}(7501:8500,j), first_USV{1,i}(7501:8500,k), '.b'); hold on;
                    plot(first_USV{1,i}(8501:10000,j), first_USV{1,i}(8501:10000,k), '.k'); hold on;
                    idxplot=idxplot+1;
                end
            end
        end
        figure; 
        idxplot=1;
        for i=1:SVD_areas
            for j = 1:3 % SVD components proj combinations (1-2, 1-3, 2-3)
                for k = j+1:3
                    subplot(SVD_areas,3,idxplot); 
                    plot(second_USV{1,i}(1:2500,j), second_USV{1,i}(1:2500,k), '.k'); hold on;
                    plot(second_USV{1,i}(2501:3500,j), second_USV{1,i}(2501:3500,k), '.r'); hold on;
                    plot(second_USV{1,i}(3501:7500,j), second_USV{1,i}(3501:7500,k), '.k'); hold on;
                    plot(second_USV{1,i}(7501:8500,j), second_USV{1,i}(7501:8500,k), '.b'); hold on;
                    plot(second_USV{1,i}(8501:10000,j), second_USV{1,i}(8501:10000,k), '.k'); hold on;
                    idxplot=idxplot+1;
                end
            end
        end
    elseif twoTonesOn 
        figure; 
        idxplot=1;
        for i=1:SVD_areas
            for j = 1:3 % SVD components proj combinations (1-2, 1-3, 2-3)
                for k = j+1:3
                    subplot(SVD_areas,3,idxplot); 
                    plot(first_USV{1,i}(intervals(1,1):intervals(1,2),j), first_USV{1,i}(intervals(1,1):intervals(1,2),k), '.k'); hold on;
                    plot(first_USV{1,i}(intervals(2,1):intervals(2,2),j), first_USV{1,i}(intervals(2,1):intervals(2,2),k), '.r'); hold on;
                    plot(first_USV{1,i}(intervals(3,1):intervals(3,2),j), first_USV{1,i}(intervals(3,1):intervals(3,2),k), '.k'); hold on;
                    plot(first_USV{1,i}(intervals(4,1):intervals(4,2),j), first_USV{1,i}(intervals(4,1):intervals(4,2),k), '.b'); hold on;
                    plot(first_USV{1,i}(intervals(5,1):intervals(5,2),j), first_USV{1,i}(intervals(5,1):intervals(5,2),k), '.k'); hold on;
                    plot(first_USV{1,i}(intervals(6,1):intervals(6,2),j), first_USV{1,i}(intervals(6,1):intervals(6,2),k), '.k'); hold on;
                    plot(first_USV{1,i}(intervals(7,1):intervals(7,2),j), first_USV{1,i}(intervals(7,1):intervals(7,2),k), '.r'); hold on;
                    plot(first_USV{1,i}(intervals(8,1):intervals(8,2),j), first_USV{1,i}(intervals(8,1):intervals(8,2),k), '.k'); hold on;
                    plot(first_USV{1,i}(intervals(9,1):intervals(9,2),j), first_USV{1,i}(intervals(9,1):intervals(9,2),k), '.b'); hold on;
                    plot(first_USV{1,i}(intervals(10,1):intervals(10,2),j), first_USV{1,i}(intervals(10,1):intervals(10,2),k), '.k'); hold on;
                    idxplot=idxplot+1;
                end
            end
        end
        figure; 
        idxplot=1;
        for i=1:SVD_areas
            for j = 1:3 % SVD components proj combinations (1-2, 1-3, 2-3)
                for k = j+1:3
                    subplot(SVD_areas,3,idxplot); 
                    plot(second_USV{1,i}(intervals(1,1):intervals(1,2),j), second_USV{1,i}(intervals(1,1):intervals(1,2),k), '.k'); hold on;
                    plot(second_USV{1,i}(intervals(2,1):intervals(2,2),j), second_USV{1,i}(intervals(2,1):intervals(2,2),k), '.r'); hold on;
                    plot(second_USV{1,i}(intervals(3,1):intervals(3,2),j), second_USV{1,i}(intervals(3,1):intervals(3,2),k), '.k'); hold on;
                    plot(second_USV{1,i}(intervals(4,1):intervals(4,2),j), second_USV{1,i}(intervals(4,1):intervals(4,2),k), '.b'); hold on;
                    plot(second_USV{1,i}(intervals(5,1):intervals(5,2),j), second_USV{1,i}(intervals(5,1):intervals(5,2),k), '.k'); hold on;
                    plot(second_USV{1,i}(intervals(6,1):intervals(6,2),j), second_USV{1,i}(intervals(6,1):intervals(6,2),k), '.k'); hold on;
                    plot(second_USV{1,i}(intervals(7,1):intervals(7,2),j), second_USV{1,i}(intervals(7,1):intervals(7,2),k), '.r'); hold on;
                    plot(second_USV{1,i}(intervals(8,1):intervals(8,2),j), second_USV{1,i}(intervals(8,1):intervals(8,2),k), '.k'); hold on;
                    plot(second_USV{1,i}(intervals(9,1):intervals(9,2),j), second_USV{1,i}(intervals(9,1):intervals(9,2),k), '.b'); hold on;
                    plot(second_USV{1,i}(intervals(10,1):intervals(10,2),j), second_USV{1,i}(intervals(10,1):intervals(10,2),k), '.k'); hold on;
                    idxplot=idxplot+1;
                end
            end
        end

    elseif thirty_kts
        figure; 
        idxplot=1;
        for i=1:SVD_areas
            for j = 1:3 % SVD components proj combinations (1-2, 1-3, 2-3)
                for k = j+1:3
                    subplot(SVD_areas,3,idxplot); 
                    plot(first_USV{1,i}(1:10000,j), first_USV{1,i}(1:10000,k), '.k'); hold on;
                    plot(first_USV{1,i}(10001:20000,j), first_USV{1,i}(10001:20000,k), '.r'); hold on;
                    plot(first_USV{1,i}(20001:30000,j), first_USV{1,i}(20001:30000,k), '.b'); hold on;
                    idxplot=idxplot+1;
                end
            end
        end
        figure; 
        idxplot=1;
        for i=1:SVD_areas
            for j = 1:3 % SVD components proj combinations (1-2, 1-3, 2-3)
                for k = j+1:3
                    subplot(SVD_areas,3,idxplot); 
                    plot(second_USV{1,i}(1:10000,j), second_USV{1,i}(1:10000,k), '.k'); hold on;
                    plot(second_USV{1,i}(10001:20000,j), second_USV{1,i}(10001:20000,k), '.r'); hold on;
                    plot(second_USV{1,i}(20001:30000,j), second_USV{1,i}(20001:300000,k), '.b'); hold on;
                    idxplot=idxplot+1;
                end
            end
        end
        
    elseif one_kts
        figure; 
        idxplot=1;
        for i=1:SVD_areas
            for j = 1:3 % SVD components proj combinations (1-2, 1-3, 2-3)
                for k = j+1:3
                    subplot(SVD_areas,3,idxplot); 
                    plot(first_USV{1,i}(1:200,j), first_USV{1,i}(1:200,k), '.r'); hold on;
                    plot(first_USV{1,i}(201:400,j), first_USV{1,i}(201:400,k), '.k'); hold on;
                    plot(first_USV{1,i}(401:600,j), first_USV{1,i}(401:600,k), '.b'); hold on;
                    plot(first_USV{1,i}(601:800,j), first_USV{1,i}(601:800,k), '.m'); hold on;
                    plot(first_USV{1,i}(801:1000,j), first_USV{1,i}(801:1000,k), '.g'); hold on;
                    idxplot=idxplot+1;
                end
            end
        end
        figure; 
        idxplot=1;
        for i=1:SVD_areas
            for j = 1:3 % SVD components proj combinations (1-2, 1-3, 2-3)
                for k = j+1:3
                    subplot(SVD_areas,3,idxplot); 
                    plot(second_USV{1,i}(1:200,j), second_USV{1,i}(1:200,k), '.r'); hold on;
                    plot(second_USV{1,i}(201:400,j), second_USV{1,i}(201:400,k), '.k'); hold on;
                    plot(second_USV{1,i}(401:600,j), second_USV{1,i}(401:600,k), '.b'); hold on;
                    plot(second_USV{1,i}(601:800,j), second_USV{1,i}(601:800,k), '.m'); hold on;
                    plot(second_USV{1,i}(801:1000,j), second_USV{1,i}(801:1000,k), '.g'); hold on;
                    idxplot=idxplot+1;
                end
            end
        end
        
    elseif nine_kts
        figure; 
        idxplot=1;
        for i=1:SVD_areas
            for j = 1:3 % SVD components proj combinations (1-2, 1-3, 2-3)
                for k = j+1:3
                    subplot(SVD_areas,3,idxplot); 
                    plot(first_USV{1,i}(1:3000,j), first_USV{1,i}(1:3000,k), '.r'); hold on;
                    plot(first_USV{1,i}(3001:6000,j), first_USV{1,i}(3001:6000,k), '.b'); hold on;
                    plot(first_USV{1,i}(6001:9000,j), first_USV{1,i}(6001:9000,k), '.g'); hold on;
                    idxplot=idxplot+1;
                end
            end
        end
        figure; 
        idxplot=1;
        for i=1:SVD_areas
            for j = 1:3 % SVD components proj combinations (1-2, 1-3, 2-3)
                for k = j+1:3
                    subplot(SVD_areas,3,idxplot); 
                    plot(second_USV{1,i}(1:3000,j), second_USV{1,i}(1:3000,k), '.r'); hold on;
                    plot(second_USV{1,i}(3001:6000,j), second_USV{1,i}(3001:6000,k), '.b'); hold on;
                    plot(second_USV{1,i}(6001:9000,j), second_USV{1,i}(6001:9000,k), '.g'); hold on;
                    idxplot=idxplot+1;
                end
            end
        end
        
    elseif six_kts
        figure; 
        idxplot=1;
        for i=1:SVD_areas
            for j = 1:3 % SVD components proj combinations (1-2, 1-3, 2-3)
                for k = j+1:3
                    subplot(SVD_areas,3,idxplot); 
                    plot(first_USV{1,i}(1:2000,j), first_USV{1,i}(1:2000,k), '.r'); hold on;
                    plot(first_USV{1,i}(2001:4000,j), first_USV{1,i}(2001:4000,k), '.b'); hold on;
                    plot(first_USV{1,i}(4001:6000,j), first_USV{1,i}(4001:6000,k), '.g'); hold on;
                    idxplot=idxplot+1;
                end
            end
        end
        figure; 
        idxplot=1;
        for i=1:SVD_areas
            for j = 1:3 % SVD components proj combinations (1-2, 1-3, 2-3)
                for k = j+1:3
                    subplot(SVD_areas,3,idxplot); 
                    plot(second_USV{1,i}(1:2000,j), second_USV{1,i}(1:2000,k), '.r'); hold on;
                    plot(second_USV{1,i}(2001:4000,j), second_USV{1,i}(2001:4000,k), '.b'); hold on;
                    plot(second_USV{1,i}(4001:6000,j), second_USV{1,i}(4001:6000,k), '.g'); hold on;
                    idxplot=idxplot+1;
                end
            end
        end
        
    else
        figure; 
        idxplot=1;
        for i=1:SVD_areas
            for j = 1:3 % SVD components proj combinations (1-2, 1-3, 2-3)
                for k = j+1:3
                    subplot(SVD_areas,3,idxplot); 
                    plot(first_USV{1,i}(1:2000,j), first_USV{1,i}(1:2000,k), '.r'); hold on;
                    plot(first_USV{1,i}(2001:4000,j), first_USV{1,i}(2001:4000,k), '.k'); hold on;
                    plot(first_USV{1,i}(4001:6000,j), first_USV{1,i}(4001:6000,k), '.b'); hold on;
                    plot(first_USV{1,i}(6001:8000,j), first_USV{1,i}(6001:8000,k), '.m'); hold on;
                    plot(first_USV{1,i}(8001:10000,j), first_USV{1,i}(8001:10000,k), '.g'); hold on;
                    idxplot=idxplot+1;
                end
            end
        end
        figure; 
        idxplot=1;
        for i=1:SVD_areas
            for j = 1:3 % SVD components proj combinations (1-2, 1-3, 2-3)
                for k = j+1:3
                    subplot(SVD_areas,3,idxplot); 
                    plot(second_USV{1,i}(1:2000,j), second_USV{1,i}(1:2000,k), '.r'); hold on;
                    plot(second_USV{1,i}(2001:4000,j), second_USV{1,i}(2001:4000,k), '.k'); hold on;
                    plot(second_USV{1,i}(4001:6000,j), second_USV{1,i}(4001:6000,k), '.b'); hold on;
                    plot(second_USV{1,i}(6001:8000,j), second_USV{1,i}(6001:8000,k), '.m'); hold on;
                    plot(second_USV{1,i}(8001:10000,j), second_USV{1,i}(8001:10000,k), '.g'); hold on;
                    idxplot=idxplot+1;
                end
            end
        end
    end
end

%% calculate SVD entropy
if entropy_bar
    H_svd = zeros(2,1);
    figure();
    for m=1:nAreas
        for n=1:Cxsz(m)
            H_svd(1) = H_svd(1) + eigenval{1,m}'*log2(eigenval{1,m});
            H_svd(2) = H_svd(2) + eigenval{2,m}'*log2(eigenval{2,m});
        end
    end
    bar(-H_svd);
end

%% show spectrum
if wvlt_spect
    figure();
    subplot(2,1,1); imagesc(abs(cwt_morlet(mean(epoch1,2), 1000, 0, 1:0.1:10, 7)));
    set(gca, 'YDir', 'normal');
    title('Time Frequency analysis - Ach OFF');
    subplot(2,1,2); imagesc(abs(cwt_morlet(mean(epoch2,2), 1000, 0, 1:0.1:10, 7)));
    set(gca, 'YDir', 'normal');
    title('Time Frequency analysis - Ach ON');
end

if fit_distr
    %% Show amplitude distribution of each SV time series
    m=3;
    figure;
    for i=1:3 % SV to display
        subplot(2,3,i);
        histogram(first_USV{1,m}(:,i));
        title(sprintf('First trace, SV%i', i));
        subplot(2,3,3+i);
        histogram(second_USV{1,m}(:,i));
        title(sprintf('Second trace, SV%i', i));
    end

    %% Fit a bimodal distribution 
    m=3; % area where tone is inserted
    nbins = 100; 
    peaks_distance = zeros(2,3);
    figure;
    for trace=1:2 % first and second USV from SVD above..
        for SV=1:3
            if trace==1
                x = first_USV{1,m}(:,SV);
            elseif trace==2
                x = second_USV{1,m}(:,SV);
            end
            pdf_normmixture = @(x,p,mu1,mu2,sigma1,sigma2) ...
                p*normpdf(x,mu1,sigma1) + (1-p)*normpdf(x,mu2,sigma2); % bimodal = mix of 2 gaussian

            pStart = .2; % mixture rate between the 2 gaussian
            muStart = quantile(x,[.25 .75]); 
            sigmaStart = sqrt(var(x) - .05*diff(muStart).^2);
            start = [pStart muStart sigmaStart sigmaStart];

            % bound of 0 and 1 for mixing probability, and lower bound of 0 for std dev 
            lb = [0 -Inf -Inf 0 0];
            ub = [1 Inf Inf Inf Inf];
            options = statset('MaxIter',400, 'MaxFunEvals',800); % more iteration to ensure convergence
            paramEsts = mle(x, 'pdf',pdf_normmixture, 'start',start, ...
                                      'lower',lb, 'upper',ub);

            subplot(2,3,3*(trace-1)+SV);
            minBin = min(x); maxBin = max(x); itvBin = (maxBin - minBin)/nbins;
            bins = minBin : itvBin : maxBin;
            %h = bar(bins,histc(x,bins)/(length(x)*.5),'histc');
            h = bar(bins(1:end-1),histcounts(x,bins, 'Normalization', 'probability'));
            h.FaceColor = [.9 .9 .9];
            xgrid = linspace(1.1*min(x),1.1*max(x), nbins);
            xfitpdf = pdf_normmixture(xgrid,paramEsts(1),paramEsts(2),paramEsts(3),paramEsts(4),paramEsts(5)); hold on;
            xfitpdf = xfitpdf/sum(xfitpdf);
            plot(xgrid,xfitpdf,'-', 'LineWidth', 2); hold off;
            xlabel('x'); ylabel('Probability Density');

            % distance between modes
            idxs = find(diff(diff(xfitpdf)<0)==1);
            if length(idxs)==2
                peaks_distance(trace,SV) = xgrid(idxs(2))-xgrid(idxs(1)); % output distance
            end
        end
    end
    % bar plot of distances between modes
    figure;
    for trace=1:2
        subplot(2,1,trace);
        bar(peaks_distance(trace,:));
    end
end




