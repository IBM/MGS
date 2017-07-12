%% Phase space trajectories movie
% Needs to be run after SORN SVD analysis in order to have SVD outputs in memory

% period to capture (epoch 1 or 2 of the SVD)
first = 0;
second = 1;

window_length = 200;  % size of the sliding window for the rasters
dt_frame = 20;         % time length displayed per frame
n_frames = (interval_length / dt_frame)-1;

fig=figure();
fig.Units = 'normalized';
%fig.Color='black';
%fig.InvertHardcopy = 'off';
fig.Position = [0.05 0.05 0.6 0.8];
fig.Name = 'Str GP Thal Ctx Activity';

vid = VideoWriter('StrGpThalCtx630kts_june01.avi');
vid.FrameRate = 10;
vid.Quality = 60;
open(vid);
% 
% strGP = subplot('Position', [0.05 0.8 0.9 0.15]); % Left, Bottom, Width, Height
% Th{1} = subplot('Position', [0.05 0.65 0.2 0.10]);
% Th{2} = subplot('Position', [0.275 0.65 0.2 0.10]);
% Th{3} = subplot('Position', [0.48 0.65 0.2 0.10]);
% Th{4} = subplot('Position', [0.725 0.65 0.2 0.10]);
% 
% ctxRaster{1} = subplot('Position', [0.05 0.5 0.2 0.1]); 
% ctxRaster{2} = subplot('Position', [0.275 0.5 0.2 0.1]); 
% ctxRaster{3} = subplot('Position', [0.48 0.5 0.2 0.1]); 
% ctxRaster{4} = subplot('Position', [0.725 0.5 0.2 0.1]);
% 
% ctxFR{1} = subplot('Position', [0.05 0.35 0.2 0.1]);
% ctxFR{2} = subplot('Position', [0.275 0.35 0.2 0.1]);
% ctxFR{3} = subplot('Position', [0.48 0.35 0.2 0.1]);
% ctxFR{4} = subplot('Position', [0.725 0.35 0.2 0.1]);
% 
% ctxTraj{1} = subplot('Position', [0.05 0.05 0.2 0.25]);
% ctxTraj{2} = subplot('Position', [0.275 0.05 0.2 0.25]);
% ctxTraj{3} = subplot('Position', [0.48 0.05 0.2 0.25]);
% ctxTraj{4} = subplot('Position', [0.725 0.05 0.2 0.25]);


strGP = subplot('Position', [0.05 0.05 0.2 0.875]); % Left, Bottom, Width, Height

Th{1} = subplot('Position', [0.3 0.725 0.2 0.2]);
Th{2} = subplot('Position', [0.3 0.5 0.2 0.2]);
Th{3} = subplot('Position', [0.3 0.275 0.2 0.2]);
Th{4} = subplot('Position', [0.3 0.05 0.2 0.2]);

ctxRaster{1} = subplot('Position', [0.55 0.83 0.2 0.095]); hold on;
ctxRaster{2} = subplot('Position', [0.55 0.61 0.2 0.095]); hold on;
ctxRaster{3} = subplot('Position', [0.55 0.38 0.2 0.095]); hold on;
ctxRaster{4} = subplot('Position', [0.55 0.155 0.2 0.095]); hold on;

ctxFR{1} = subplot('Position', [0.55 0.725 0.2 0.095]); hold on;
ctxFR{2} = subplot('Position', [0.55 0.5    0.2 0.095]); hold on;
ctxFR{3} = subplot('Position', [0.55 0.275 0.2 0.095]); hold on;
ctxFR{4} = subplot('Position', [0.55 0.05 0.2 0.095]); hold on;

ctxTraj{1} = subplot('Position', [0.8 0.725 0.2 0.2]); %hold on;
ctxTraj{2} = subplot('Position', [0.8 0.5 0.2 0.2]); %hold on;
ctxTraj{3} = subplot('Position', [0.8 0.275 0.2 0.2]); %hold on;
ctxTraj{4} = subplot('Position', [0.8 0.05 0.2 0.2]); %hold on;

for m=1:nAreas
    axes(ctxTraj{m});
    p_trail{m} = plot3(USV{1,m}(1,comp(1)), USV{1,m}(1,comp(2)), USV{1,m}(1,comp(3)), 'o');
    p_trail{m}.MarkerFaceColor='none'; 
    p_trail{m}.MarkerEdgeColor='none'; 
    p_trail{m}.MarkerSize=1; hold on;
    set(gca, 'XTick', []); xlabel('SV1');
    set(gca, 'YTick', []); ylabel('SV2');
    set(gca, 'ZTick', []); zlabel('SV3');
    xmin = min(USV{1,m}(:,comp(1))); 
    xmax = max(USV{1,m}(:,comp(1)));
    ymin = min(USV{1,m}(:,comp(2))); 
    ymax = max(USV{1,m}(:,comp(2)));
    zmin = min(USV{1,m}(:,comp(3))); 
    zmax = max(USV{1,m}(:,comp(3)));
    xlim([xmin xmax]);
    ylim([ymin ymax]);
    zlim([zmin zmax]);
    hold on;
end

strgp = zeros(interval_length+1,nStr);
if first
    USV = first_USV;
    t_start = t1;
    epoch = epoch1;
    strgp(:,D1) = hx_s(t_start:t_start+interval_length,D1)>0;
    strgp(:,D2) = -(hx_s(t_start:t_start+interval_length,D2)>0);
    Thal = x_s(t_start:t_start+interval_length,:);
elseif second
    USV = second_USV;
    t_start = t3;
    epoch = epoch2;
    strgp(:,D1) = hx_s(t_start:t_start+interval_length,D1)>0;
    strgp(:,D2) = -(hx_s(t_start:t_start+interval_length,D2)>0);
    Thal = x_s(t_start:t_start+interval_length,:);
end
%Thal = Thal - mean(Thal);

tail_length = ceil(dt_frame/2);
body_length = ceil(dt_frame/4);
head_length = ceil(dt_frame/5);

blck = [0 0 0];
dark_gray = [0.2 0.2 0.2];
medium_dark_gray = [0.4 0.4 0.4];
medium_light_gray = [0.6 0.6 0.6];
light_gray = [0.8 0.8 0.8];
wht = [1 1 1];

blue = [0 0 1];
dark_blue = [0 0 0.2];
medium_dark_blue = [0 0 0.4];
medium_light_blue = [0 0 0.6];
light_blue = [0 0 0.8];

red = [1 0 0];
dark_red = [0.2 0 0];
medium_dark_red = [0.4 0 0];
medium_light_red = [0.6 0 0];
light_red = [0.8 0 0];

green = [0 1 0];
dark_green = [0 0.2 0];
medium_dark_green = [0 0.4 0];
medium_light_green = [0 0.6 0];
light_green = [0 0.8 0];


for i=333:n_frames
    %--- StrGP ---%
    axes(strGP)
    if i < window_length/dt_frame
        hold off;
        imagesc(strgp(1:i*dt_frame,:)');
        xlim([0 window_length]);
    elseif i == window_length/dt_frame
        imagesc(strgp(1:interval_length,:)');
        xlim([0 window_length]);
        hold on;
    else
        xlim([(i-1)*dt_frame-window_length+1 (i-1)*dt_frame]);
    end
    colormap(strGP, cbrewer('div', 'PuOr', 11)); 
    caxis([-1.6 1.6]);
    xlabel('time (ms)'); ylabel('Units')
    title('GP units up-states');
    
    for m=1:nAreas
        %--- Thalamus ---%
        axes(Th{m});
        if i < window_length/dt_frame
            hold off;
            imagesc(Thal(1:i*dt_frame,Cxoffs(m)+1:Cxoffs(m+1))');
            xlim([0 window_length]);
        elseif i == window_length/dt_frame
            hold on;
            imagesc(Thal(:,Cxoffs(m)+1:Cxoffs(m+1))');
            xlim([0 window_length]);
        else
            xlim([(i-1)*dt_frame-window_length+1 (i-1)*dt_frame]);
        end 
        clrmp = cbrewer('div', 'RdBu', 11);
        colormap(Th{m}, flipud(clrmp));
        axlim = max(abs(min(min(Thal(:,Cxoffs(m)+1:Cxoffs(m+1))))), max(max(Thal(:,Cxoffs(m)+1:Cxoffs(m+1)))));
        caxis([-axlim axlim]);
        if (m==4) xlabel('time (ms)'); end;
        ylabel('Units')
        title(sprintf('Thalamic Area %i firing rates', m));
        

        %--- Spiking ---% 
        axes(ctxRaster{m}); 
        if i < window_length/dt_frame
            hold off;
            imagesc(SFRE_raster(t_start:t_start+i*dt_frame,Cxoffs(m)+1:Cxoffs(m+1))');
            xlim([0 window_length]);
        elseif i == window_length/dt_frame
            imagesc(SFRE_raster(t_start:t_start+interval_length,Cxoffs(m)+1:Cxoffs(m+1))');
            xlim([0 window_length]);
            hold on;
        else
            xlim([(i-1)*dt_frame-window_length+1 (i-1)*dt_frame]);
        end
        colormap(ctxRaster{m}, flipud(gray));
        set(gca, 'XTickLabel', []); 
        ylabel('Units')
        title(sprintf('Cortical Area %i spiking',m));

        %--- Firing Rate ---%
        axes(ctxFR{m});
        if i < window_length/dt_frame
            hold off;
            imagesc(epoch(1:i*dt_frame,Cxoffs(m)+1:Cxoffs(m+1))');
            xlim([0 window_length]);
        elseif i == window_length/dt_frame
            hold on;
            imagesc(epoch(:,Cxoffs(m)+1:Cxoffs(m+1))');
            xlim([0 window_length]);
        else
            xlim([(i-1)*dt_frame-window_length+1 (i-1)*dt_frame]);
        end 
        clrmp = cbrewer('div', 'PiYG', 11);
        %clrmp(1:6,:) = 1;
        colormap(ctxFR{m}, clrmp);
        caxis([-0.1 0.3]);
        if m==4 xlabel('time (ms)'); 
        else set(gca, 'XTickLabel', []); 
        end
        ylabel('Units')
        title(sprintf('Cortical Area %i firing rates',m));

        %--- Trajectory ---%
        axes(ctxTraj{m}); hold on;
        if i>1
            delete(p_body{m});
            delete(p_head{m});
            delete(p_tail{m});
        end
            
        % used for gradation of white to black
        rnge_4 = (i-1)*dt_frame + 1                         :   i*dt_frame;                                            % "trail"
        rnge_3 = i*dt_frame + 1                             :   i*dt_frame + tail_length;                             % "tail"
        rnge_2 = i*dt_frame + tail_length + 1               :   i*dt_frame + tail_length + body_length;               % "body"
        rnge_1 = i*dt_frame + tail_length + body_length + 1 :   i*dt_frame + tail_length + body_length + head_length; % "head"
        
        if i*dt_frame < round(interval_length/3)
            trail_color = medium_dark_red;
            tail_color = medium_light_red;
            body_color = light_red;
            head_color = red;
        elseif i*dt_frame < 2*(interval_length/3)
            trail_color = medium_dark_green;
            tail_color = medium_light_green;
            body_color = light_green;
            head_color = green;
        else
            trail_color = medium_dark_blue;
            tail_color = medium_light_blue;
            body_color = light_blue;
            head_color = blue;
        end
        
        % trail  
        if i>1
            p_trail{m} = plot3(USV{1,m}(rnge_4,comp(1)), USV{1,m}(rnge_4,comp(2)), USV{1,m}(rnge_4,comp(3)), 'o');
            p_trail{m}.MarkerFaceColor=trail_color; p_trail{m}.MarkerEdgeColor='none'; p_trail{m}.MarkerSize=2; hold on;
        end

        % tail
        p_tail{m} = plot3(USV{1,m}(rnge_3,comp(1)), USV{1,m}(rnge_3,comp(2)), USV{1,m}(rnge_3,comp(3)),'o');
        p_tail{m}.MarkerFaceColor=tail_color; p_tail{m}.MarkerEdgeColor='none'; p_tail{m}.MarkerSize=3; hold on;

        % body
        p_body{m} = plot3(USV{1,m}(rnge_2,comp(1)), USV{1,m}(rnge_2,comp(2)), USV{1,m}(rnge_2,comp(3)),'o'); 
        p_body{m}.MarkerFaceColor=body_color; p_body{m}.MarkerEdgeColor='none'; p_body{m}.MarkerSize=4; hold on;

        % head
        p_head{m} = plot3(USV{1,m}(rnge_1,comp(1)), USV{1,m}(rnge_1,comp(2)), USV{1,m}(rnge_1,comp(3)),'o'); 
        p_head{m}.MarkerFaceColor=head_color; p_head{m}.MarkerEdgeColor='none'; p_head{m}.MarkerSize=5; hold on;

        %axis('off');
        title(sprintf('Cortical Area %i trajectory',m));
    end
    
    %print(sprintf('SpikeRateTraj%03i', i), '-dpng');
    frame = getframe(fig);
    writeVideo(vid, frame);
    if (mod(i,100)==0) display(i); end;
end

close(vid);

