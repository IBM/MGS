clear variables; close all;
set(0,'defaulttextinterpreter','tex'); rng('shuffle');
%% Parameters
figNum=0;
% general parameters
sf=0.1;
% sf=0.01;
directory='../../graphs/SpinalCord/';
fileExt='.dat';
% time frames
T=sf:sf:1000; % Length of simulation time saving data in ms to load and process
% T=sf:sf:10000;
measureHz=500:sf:T(end); % When to measure firing rates for IF curve(s) in msword
%% Whether to load/process different data sources
postprocess_compartments = [true, true, true];%, true]; % dendrite, soma, IAS, axon
compartments = [{'d_'}, {'s_'}, {'i_'}];%, {'a_'}];
compartmentsStr_V_m = [{'Dendrite (proximal)'}, {'Soma '}, {'IAS'}];%, {'Axon (last node)'}];
compartmentsStr_I_in = [{'Dendrite (distal)'}, {'Soma '}];
postprocess_V_m = true;
postprocess_I_in = true;
postprocess_everythingElse = true;
postprocess_IF = false;
% Load Data
if (postprocess_V_m)
    V_m = cell(1,4);
    for c=1:numel(postprocess_compartments)
        if (postprocess_compartments(c))
            if (c==1)
                [V_m{c}, Xdim, Ydim, Zdim] = load1D(directory, ...
                    [compartments{c}, 'V_m_last'], fileExt, 'double'); % N.B. only the last node
            elseif (c==4)
                [V_m{c}, Xdim, Ydim, Zdim] = load1D(directory, ...
                    [compartments{c}, 'V_m_node_last'], fileExt, 'double'); % N.B. only the last node
            else
                [V_m{c}, Xdim, Ydim, Zdim] = load1D(directory, ...
                    [compartments{c}, 'V_m'], fileExt, 'double');
            end
            % 4D: time, X, Y, Z
            V_m{c} = reshape(V_m{c}, [Xdim, Ydim, Zdim, round(T(end)/sf)]);
            V_m{c} = permute(V_m{c}, [4, 1, 2, 3]);
        end
    end
end
if (postprocess_I_in)
    I_in = cell(1,3);
    for c=1:2 % only dendrite and soma have input
        if (postprocess_compartments(c))
            [I_in{c}, Xdim, Ydim, Zdim] = load1D(directory, ...
                [compartments{c}, 'I_in'], fileExt, 'double');
            % 4D: time, X, Y, Z
            I_in{c} = reshape(I_in{c}, [Xdim, Ydim, Zdim, round(T(end)/sf)]);
            I_in{c} = permute(I_in{c}, [4, 1, 2, 3]);
        end
    end
end
if (postprocess_everythingElse)
    everythingElse = cell(1,3);
    for c=1:numel(postprocess_compartments)
        if (postprocess_compartments(c))
            [everythingElse{c}, Xdim, Ydim, Zdim] = load1D(directory, ...
                [compartments{c}, 'everythingElse'], ...
                fileExt, 'double');
            switch c
                case 1
                    % DENDRITE
                    % NONE!
                case 2
                    % SOMA
                    % 4D: time, X, Y, Z, [1 = I_Naf
                    %                     2 = I_Kdr
                    %                     3 = I_CaN
                    %                     4 = I_CaL
                    %                     5 = I_KCa
                    %                     6 = I_leak
                    %                     7 = m_Naf
                    %                     8 = h_Naf
                    %                     9 = n_Kdr
                    %                     10 = m_CaN
                    %                     11 = h_CaN
                    %                     12 = p_CaL
                    %                     13 = Ca_i
                    %                     14 = E_Ca]
                    everythingElse{c} = reshape(everythingElse{c}, [14, Xdim, Ydim, Zdim, ...
                        round(T(end)/sf)]);
                    everythingElse{c} = permute(everythingElse{c}, [5,2,3,4,1]);
                case 3
                    % AIS
                    % 4D: time, X, Y, Z, [1 = I_Naf
                    %                     2 = I_Nap
                    %                     3 = I_Kdr
                    %                     4 = I_leak
                    %                     5 = m_Naf
                    %                     6 = h_Naf
                    %                     7 = p_Nap
                    %                     8 = n_Kdr]
                    everythingElse{c} = reshape(everythingElse{c}, [8, Xdim, Ydim, Zdim, ...
                        round(T(end)/sf)]);
                    everythingElse{c} = permute(everythingElse{c}, [5,2,3,4,1]);
                case 4
                    % Axon
                    % 4D: time, X, Y, Z, [1 = I_Naf_last
                    %                     2 = I_Nap_last
                    %                     3 = I_Ks_last
                    %                     4 = I_leak_last
                    %                     5 = m_Naf_last
                    %                     6 = h_Naf_last
                    %                     7 = p_Nap_last
                    %                     8 = s_Ks_last]
                    everythingElse{c} = reshape(everythingElse{c}, [8, Xdim, Ydim, Zdim, ...
                        round(T(end)/sf)]);
                    everythingElse{c} = permute(everythingElse{c}, [5,2,3,4,1]);
            end            
        end
    end
end
% Plot
neuron=randi(Xdim,1,1); % only for Xdimension
if (postprocess_V_m || postprocess_I_in)
    [~, figNum] = newFigure(figNum, true); % sample V_m and I_in
    subplot(9,1,1:4); hold on;
    subplot(9,1,8:9); hold on;
    for c=1:numel(postprocess_compartments)
        if (postprocess_compartments(c))    
            if (postprocess_V_m)
                if (c > 1)
                    subplot(9,1,1:4);
                else
                    subplot(9,1,5:6);
                end
                plot(T/1000, V_m{c}(:,neuron,1,1), ...
                    'DisplayName', compartmentsStr_V_m{c});
                if (c > 1)
                    title('V_{m}');
                end
                xlabel('t [s]');
                ylabel('[mV]');
            end
            if (postprocess_I_in)
                if (c <= 2) % only dendrite and soma have input
                    subplot(9,1,8:9);
                    plot(T/1000, -I_in{c}(:,neuron,1,1)*1000000, ...
                        'DisplayName', compartmentsStr_I_in{c});
                    title('I_{in}');
                    ylabel('[nA]');
                    xlabel('t [s]');
                end
            end
        end
    end
    subplot(9,1,1:4); hold off; legend('show');
    subplot(9,1,5:6); legend('show');
    subplot(9,1,8:9); hold off; legend('show');
    print([directory,'V_m-I_in'],'-dpng');
end
if (postprocess_everythingElse)
    for c=1:numel(postprocess_compartments)
        if (postprocess_compartments(c))
            switch c
                case 1
                    % DENDRITE
                    % NONE!
                case 2
                    % SOMA
                    [~, figNum] = newFigure(figNum, true);  
                    % sample I_Naf
                    subplot(2,4,1);
                    hold on;
                    a1 = plot(everythingElse{c}(:,neuron,1,1,1)); m1 = 'I_{Naf}';
                    yyaxis right;
                    a3 = plot(everythingElse{c}(:,neuron,1,1,8)); m3 = 'h_{Naf}';
                    a2 = plot(everythingElse{c}(:,neuron,1,1,7)); m2 = 'm_{Naf}';
                    ylim([0 1]);
                    legend([a1; a2; a3], [m1; m2; m3]);
                    title('Fast sodium current');
                    xlabel('t [dt]');
                    % sample I_Kdr
                    subplot(2,4,2);  
                    hold on;
                    a1 = plot(everythingElse{c}(:,neuron,1,1,2)); m1 = 'I_{Kdr}';
                    yyaxis right;
                    a2 = plot(everythingElse{c}(:,neuron,1,1,9)); m2 = 'n_{Kdr}';
                    legend([a1; a2], [m1; m2]);
                    title('Delayed rectifier potassium current');
                    xlabel('t [dt]');
                    % sample I_CaN
                    subplot(2,4,3);    
                    hold on;
                    a1 = plot(everythingElse{c}(:,neuron,1,1,3)); m1 = 'I_{CaN}';
                    yyaxis right;
                    a2 = plot(everythingElse{c}(:,neuron,1,1,10)); m2 = 'm_{CaN}';
                    a3 = plot(everythingElse{c}(:,neuron,1,1,11)); m3 = 'h_{CaN}';
                    ylim([0 1]);
                    legend([a1; a2; a3], [m1; m2; m3]);
                    title('N-type calcium current');
                    xlabel('t [dt]');
                    % sample I_CaL
                    subplot(2,4,4);   
                    hold on;
                    a1 = plot(everythingElse{c}(:,neuron,1,1,4)); m1 = 'I_{CaL}';
                    yyaxis right;
                    a2 = plot(everythingElse{c}(:,neuron,1,1,12)); m2 = 'p_{CaL}';
                    legend([a1; a2], [m1; m2]);
                    title('L-type calcium current');
                    xlabel('t [dt]');
                    % sample I_KCa
                    subplot(2,4,5);  
                    hold on;
                    a1 = plot(everythingElse{c}(:,neuron,1,1,5)); m1 = 'I_{KCa}';
                    legend([a1], [m1]);
                    title('Calcium-activated potassium current');
                    xlabel('t [dt]');
                    % sample I_leak
                    subplot(2,4,6); 
                    hold on;
                    a1 = plot(everythingElse{c}(:,neuron,1,1,6)); m1 = 'I_{leak}';
                    legend([a1], [m1]);
                    title('Leak');
                    xlabel('t [dt]');
                    % Calcium dynamics
                    subplot(2,4,7);  
                    hold on;
                    a1 = plot(everythingElse{c}(:,neuron,1,1,13)); m1 = 'Ca_{i}';
                    yyaxis right;
                    a2 = plot(everythingElse{c}(:,neuron,1,1,14)); m2 = 'E_{Ca}';
                    legend([a1; a2], [m1; m2]);
                    title('Calcium dynamics');
                    xlabel('t [dt]');

                    print([directory,'soma_everythingElse'],'-dpng');
                case 3
                    % AIS
                    [~, figNum] = newFigure(figNum, true);  
                    % sample I_Naf
                    subplot(2,2,1);
                    hold on;
                    a1 = plot(everythingElse{c}(:,neuron,1,1,1)); m1 = 'I_{Naf}';
                    yyaxis right;
                    a3 = plot(everythingElse{c}(:,neuron,1,1,5)); m3 = 'm_{Naf}';
                    a2 = plot(everythingElse{c}(:,neuron,1,1,6)); m2 = 'h_{Naf}';
                    ylim([0 1]);
                    legend([a1; a2; a3], [m1; m2; m3]);
                    title('Fast sodium current');
                    xlabel('t [dt]');
                    % sample I_Nap
                    subplot(2,2,2);  
                    hold on;
                    a1 = plot(everythingElse{c}(:,neuron,1,1,2)); m1 = 'I_{Nap}';
                    yyaxis right;
                    a2 = plot(everythingElse{c}(:,neuron,1,1,7)); m2 = 'p_{Nap}';
                    legend([a1; a2], [m1; m2]);
                    title('Persistent sodium current');
                    xlabel('t [dt]');
                    % sample I_Kdr
                    subplot(2,2,3);    
                    hold on;
                    a1 = plot(everythingElse{c}(:,neuron,1,1,3)); m1 = 'I_{Kdr}';
                    yyaxis right;
                    a2 = plot(everythingElse{c}(:,neuron,1,1,8)); m2 = 'n_{Kdr}';
                    legend([a1; a2], [m1; m2]);
                    title('Delayed rectifier potassium current');
                    xlabel('t [dt]');
                    % sample I_leak
                    subplot(2,2,4);   
                    hold on;
                    a1 = plot(everythingElse{c}(:,neuron,1,1,4)); m1 = 'I_{leak}';
                    legend([a1], [m1]);
                    title('Leak');
                    xlabel('t [dt]');

                    print([directory,'AIS_everythingElse'],'-dpng');
                case 4
                    % AXON
                    [~, figNum] = newFigure(figNum, true);  
                    % sample I_Naf
                    subplot(2,2,1);
                    hold on;
                    a1 = plot(everythingElse{c}(:,neuron,1,1,1)); m1 = 'I_{Naf}';
                    yyaxis right;
                    a3 = plot(everythingElse{c}(:,neuron,1,1,5)); m3 = 'm_{Naf}';
                    a2 = plot(everythingElse{c}(:,neuron,1,1,6)); m2 = 'h_{Naf}';
                    ylim([0 1]);
                    legend([a1; a2; a3], [m1; m2; m3]);
                    title('Fast sodium current');
                    xlabel('t [dt]');
                    % sample I_Nap
                    subplot(2,2,2);  
                    hold on;
                    a1 = plot(everythingElse{c}(:,neuron,1,1,2)); m1 = 'I_{Nap}';
                    yyaxis right;
                    a2 = plot(everythingElse{c}(:,neuron,1,1,7)); m2 = 'p_{Nap}';
                    legend([a1; a2], [m1; m2]);
                    title('Persistent sodium current');
                    xlabel('t [dt]');
                    % sample I_Kdr
                    subplot(2,2,3);    
                    hold on;
                    a1 = plot(everythingElse{c}(:,neuron,1,1,3)); m1 = 'I_{Ks}';
                    legend([a1], [m1]);
                    title('Slow potassium current');
                    xlabel('t [dt]');
                    % sample I_leak
                    subplot(2,2,4);   
                    hold on;
                    a1 = plot(everythingElse{c}(:,neuron,1,1,4)); m1 = 'I_{leak}';
                    legend([a1], [m1]);
                    title('Leak');
                    xlabel('t [dt]');

                    print([directory,'axon_everythingElse'],'-dpng');
            end
        end
    end
end
if (postprocess_IF)
    [~, figNum] = newFigure(figNum, false); % IF curve
    if (postprocess_compartments(2)) % measures spikes in the AXON and considers input in the SOMA
        i=1;
        Hz_raw = zeros(Xdim*Ydim*Zdim,1);
        Hz_inter = zeros(Xdim*Ydim*Zdim,1);
        I_inStart = zeros(Xdim*Ydim*Zdim,1);
        for x=1:Xdim
            for y=1:Ydim
                for z=1:Zdim
                    [peaks, locs] = findpeaks(V_m{4}(:,x,y,z));
                    temp = locs(peaks >= 0 & locs >= (measureHz(1)/sf) ... 
                        & locs <= (measureHz(end)/sf));
                    Hz_raw(i) = numel(temp)*(1000/(measureHz(end)-measureHz(1)));
                    if (numel(temp)>=2)
                        Hz_inter(i) = (1000/sf)/(temp(end)-temp(end-1));
                    else
                        Hz_inter(i) = 0;
                    end
                    I_inStart(i) = I_in{2}(1,x,y,z); % assuming constant
                    i=i+1;  
                    clear peaks locs temp;
                end
            end
        end
        subplot(2,1,1); hold on;
        scatter(-I_inStart*1000000, Hz_raw, ...
            'DisplayName', compartmentsStr_V_m{4});
        xlabel('I_in [nA]'); ylabel('Hz');
        legend show;
        subplot(2,1,2); hold on;
        scatter(-I_inStart*1000000, Hz_inter, ...
            'DisplayName', compartmentsStr_V_m{4});
        xlabel('I_in [nA]'); ylabel('Hz');
        legend show;
        print([directory,'IF'],'-dpng');
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
function [ret, Xdim, Ydim, Zdim] = load1D(directory, file, fileExt, type)
    fid = fopen([directory,file,fileExt],'r');
    Xdim = fread(fid, 1, 'int');
    Ydim = fread(fid, 1, 'int');
    Zdim = fread(fid, 1, 'int');
    ret = fread(fid, Inf, type);
    fclose(fid);
end