%% Load SORN Ring, and trained IM features
rng(92524);
start = 1;
load(['SFR_200000.mat']);
A_0=zeros(NE);
A_0(1:55,147:238)=1;
A_0(56:146,1:55)=1;
A_0(147:238,239:400)=1;
A_0(239:400,56:146)=1;

%% IBEX Configuration variables
niters = 1000000;
sz = niters-start+1;
t=0.0;

nDA=20;         % number of Mihalas-Niebur IAF rebound bursting Dopamine neurons
nStr=100;       % number of Fitzhugh-Nagumo winnerless Striatal neurons
nStr2DA=20;     % number of Striatal neurons projecting to a Dopamine neuron
Cxsz = diff(bnd);           % size of each cortical area self organized from SFR_200000.mat
Cxoffs = [0 cumsum(Cxsz)];
nCx = sum(Cxsz);            % total size of cortex
nAreas =  size(Cxsz,2);     % number of cortical areas
CxFF = cell(nAreas,1);      % array of weight matrices for inverting the SORN FB ring
CxFB = cell(nAreas,1);      % array of weight matrices for recapitulating the SORN FB ring
nCx2Str = 20;               % number of cortical neurons projecting to a Striatal neuron
nStr2Th = 3;               % number of striatal neurons projecting to a Thalamic neuron       /!\ deprecated if topographic_StrTh=1, then use D1_center & D2_surround instead 
nFrontal = 2;               % number of frontal lobe areas
CxStr = 0;                  % 0:opened loop;  1: closed loop 
Strsz = zeros(nFrontal,1); %striatum is separated in the same amount of regions than there are frontal lobes
% but the size of str regions affected to a frontal area is not
% proportional to the number of neurons contains in this area

D2_only = 0;  % flag for StrTh gating type
eta_STDP = 4*10^-2;
eta_inhib = 10^-2;
eta_IP = 10^-3;
%HIPp=0.1;

topographic_StrTh = 1; % need Cxsz(m)> nStr/2 
D2_surround = 8;
D1_center = 16;

 
remStr = nStr; %rem = remaining str neurons to be affected
for m=1:nFrontal
    Strsz(m)=round(remStr/(nFrontal-m+1));
    remStr=remStr-Strsz(m);
end
Stroffs = [0; cumsum(Strsz)];
Cx2StrAreas = [1 2];

SFRE_raster = zeros(sz, NE);
SFRI_raster = zeros(sz, NI);

%% Mihalas-Niebur DA variables

b = 1.0;                 % s^-1
GoC = 50;                % s^-1
k(1) = 200;              % s^-1
k(2) = 20;               % s^-1
Theta_inf = -0.05;       % V
R(1) = 0;
R(2) = 1.0;
E_L = -0.07;             % V
V_r = -0.07;             % V
Theta_r = -0.06;         % V
a = 1.0;                 % s^-1
An(1) = 5.0;             % V/s - 'A(1)' from DA.m
An(2) = -0.3;            % V/s - 'A(2)' from DA.m
nI = size(An,2);            % number of intrinsic currents
dt=0.001;                   % time step
I = zeros(sz, nI, nDA);     % intrinsic currents
I_e = zeros(sz, nDA);       % extrinsic current
V = zeros(sz, nDA);
DA_raster = zeros(sz, nDA);
Theta = zeros(sz, nDA);
V_0 = ones(1,nDA) * V_r;              % V
Theta_0 = ones(1,nDA) * Theta_inf;    % V
V(1,:) = V_0;
Theta(1,:) = Theta_0;
I_p = zeros(2, nI, nDA);    % numerical intermediates for Runga-Kutta iteration
V_p = ones(2, nDA);
Theta_p = ones(2, nDA);
for ip=1:2
    V_p(ip,:)=V_p(ip,:).*V_0;
    Theta_p(ip,:)=Theta_p(ip,:).*Theta_0;
end
Cap = 1.0;                  % capacitance; 'C' from DA.m
G = GoC*Cap;
np=1;                       % number of Runga-Kutta iterations
tauDA = 25;                 % number of time steps Dopamine persists in synaptic space
DAsp = zeros(tauDA, nDA);

%% FitzHugh-Nagumo Str variables
xfn=zeros(sz,nStr);          % x from winnerless.m
yfn=zeros(sz,nStr);          % y from winnerless.m
zfn=zeros(sz,nStr);          % z from winnerless.m
g=zeros(nStr,1);        % positive x (i.e x.*(x>0)) 
hx=zeros(1,nStr)+0.5;
hx_s = zeros(niters,nStr);
hxprev=zeros(1,nStr);
hxsp=zeros(1,nStr);
hxspprev=zeros(1,nStr);
LI=zeros(nStr,nStr);        % Str local connectivity matrix
LImax=0.25;
xfn(1,:)=-1.2;
yfn(1,:)=-0.62;
zfn(1,:)=0;
GABAtD1=0.00707;            % Threshold for GABAup in D1
GABAtD2=0.00707;            % Threshold for GABAup in D2
eta_CxStr=0.0020;           % CxStr learning rate

CxStrW0 = 0.1/nCx2Str;      % Initial CxStr input weighting
step=0.01;                  % Str spikes modeled as 100 msec GABA-A potentials 
step1 = step/0.04; %0.08;
step2 = step/0.5;
step3 = step/2.05; %//4.1;
NU=-1.5;
afn = 0.7;                  % 'a' in winnerless.m
bfn = 0.8;                  % 'b' in winnerless.m
IN = zeros(nStr,sz);
bidir = 1;             % set to 1 if inh-inh bidirectional permitted
gamma_kernel = 1; 
asym = 0;               % set to 1 if gamma kernel connectivity should be asymetric
plot_ckernel = 1;
maxIN=4.0;

Ic = 0.325;          % constant input intensity
Ir = 0.025;          % intensity of (time varying) random input
Pr = 100;            % period between changing value of random input
P = zeros(1,nStr);
sigmaP=0.025;        % std dev of normally distributed random input 
npatts=6;
stim_start = 600000;
stim_end = 800000;
stim_dur = 500;     % used in post-stim only
stim_intv = 3333;   %
stim_max = 0.15;
% Spatially clustered stimulus
B=zeros(npatts, nStr); 
if npatts==4
    B(1,1:25) = normrnd(stim_max, 0.05,[1,25]);
    B(2,26:50) = normrnd(stim_max, 0.05,[1,25]);
    B(3,51:75) = normrnd(stim_max, 0.05,[1,25]);
    B(4,76:100) = normrnd(stim_max, 0.05,[1,25]);
elseif npatts==6
    B(1,1:16) = normrnd(stim_max, 0.05,[1,16]);
    B(2,17:33) = normrnd(stim_max, 0.05,[1,17]);
    B(3,34:50) = normrnd(stim_max, 0.05,[1,17]);
    B(4,51:66) = normrnd(stim_max, 0.05,[1,16]);
    B(5,67:83) = normrnd(stim_max, 0.05,[1,17]);
    B(6,84:100) = normrnd(stim_max, 0.05,[1,17]);
elseif npatts==10
    B(1,1:10) = normrnd(stim_max, 0.05,[1,10]);
    B(2,11:20) = normrnd(stim_max, 0.05,[1,10]);
    B(3,21:30) = normrnd(stim_max, 0.05,[1,10]);
    B(4,31:40) = normrnd(stim_max, 0.05,[1,10]);
    B(5,41:50) = normrnd(stim_max, 0.05,[1,10]);
    B(6,51:60) = normrnd(stim_max, 0.05,[1,10]);
    B(7,61:70) = normrnd(stim_max, 0.05,[1,10]);
    B(8,71:80) = normrnd(stim_max, 0.05,[1,10]);
    B(9,81:90) = normrnd(stim_max, 0.05,[1,10]);
    B(10,91:100) = normrnd(stim_max, 0.05,[1,10]);
end
    
% B(3,41:60) = normrnd(stim_max, 0.05,[1,20]);
% B(4,61:80) = normrnd(stim_max, 0.05,[1,20]);
% B(5,81:100) = normrnd(stim_max, 0.05,[1,20]);
% Stimulus vector (to be updated later in simulation)
S = zeros(1,nStr);

stim_noise=1;   %  1: spatial random pattern of stimulus is changed at every stim

%% Infomax L2/3 variables
learning_rate = 0.0007;
invt = 10;                                     % learning interval of matrix inversion for efficiency
Xsize = 50000;                                  % estimated number of samples in one pass through input space
bias = cell(nAreas,1);
w = cell(nAreas,1);
q = cell(nAreas,1);                     %covariance matrix to calculate whitening factor
c = cell(nAreas,1);                     %whitening vector
w0 = cell(nAreas,1);
delta_w = cell(nAreas,1);
delta_w0 = cell(nAreas,1);
usize = Cxsz;                                   % length of the linear output vector
sg = zeros(sz, NE);                             % supragranular response
Ach = 0.0;                                     % max Infomax sg gating. Value of Ach until t_Ach is reached
Ach_2 = 1.0;                    %value of Ach once from t_Ach onwards
t_Ach = 200000;  %time at which Ach feedback is turned on
tauX = 500;                                     % integration window size of L2/3
xx = cell(nAreas,1);
x_s = zeros(niters,NE);  %storage of infomax input
y_s = zeros(niters,NE);  %storage of infomax output after passing the logistic function
u_s = zeros(niters,NE);  %storage of infomax output
sg(1,:) = abs(randn(1,NE));
sg(1,:) = 1.0 - (Ach * (sg(1,:)/max(sg(1,:))));
for m=1:nAreas
    bias{m} = zeros( Cxsz(m), 1);               % input bias vector (ensures zero-mean inputs)
    xx{m} = zeros( Cxsz(m), tauX);
    w{m} = randn( Cxsz(m) );                    % the input weight matrix (undergoes infomax learning)
    normw = sqrt(sum((w{m}.*w{m})'));			% normalization factor
    for n=1:Cxsz(m)                             % normalize input weights, yields weights uniformly distributed on the unit sphere
        w{m}(n,:) = w{m}(n,:)./normw(n);
    end
    w0{m} = zeros(usize(m),1);                  % bias weights
    delta_w{m} = zeros(usize(m));               % accumulated partial input weights learning vector
    delta_w0{m} = zeros(usize(m), 1);           % accumulated bias weights learning vector
    q{m} = zeros(usize(m));
    c{m} = zeros(usize(m));
end

%% Backup variables
%infomax bckup
Q = cell(nAreas,1);                             % correlation matrix of yy
checkpoint = 10000;
backups = ceil((niters-start)/checkpoint)+1;    % computes the number of backups made based on the number of training iterations and the checkpoint interval
backup = 1;										% backup counter
wb=cell(nAreas, backups);                       % the array of weight matrices stored during learning (1 every "checkpoint" iterations)
cb=cell(nAreas, backups);
for m=1:nAreas
    wb{m, backup} = w{m}; 						% wb{m,1} is weights before learning
    Q{m} = zeros(usize(m));                     % accumulated partial input weights learning vector
    cb{m, backup} = c{m};
end

%SFR bckup
weeb=zeros(backups,NE,NE);
weeb(1,:,:)=WEEp;

backup = backup + 1;							% increment backup counter

%% IBEX Configuration

% Randomly assign Striatal inputs to DA neurons
fid=zeros(nDA,nStr);
for j=1:nDA
    [rn fid(j,:)]=sort(rand(1,nStr));
end
fid=fid(:,1:nStr2DA);  % fid : Str2DA indexes

%Unitary input current for Str to DA input
I_StrDA = -2.25;

% Set up connections within Str winnerless network
LI = (rand(size(LI))<0.35) .* ~eye(size(LI)) .* rand(size(LI)) * LImax;

% Connectivity kernel
if gamma_kernel
    % gamma distribution
    offset=round(nStr/2);
    x_=-offset+1:offset;
    theta=5; % scale
    n=5;     % shape
    W=(x_.^(n-1).*exp(-abs(x_)/theta))/(2*theta^n*factorial(n-1));
    if asym
        W(offset:end) = W(offset:end)/4; 
    end

    %% Gamma distribution 
    if(plot_ckernel)
        figure;  % for visualizing kernel in 1D
        subplot(1,2,1);
        for theta=5
            for n=5
                W = (x_.^(n-1).*exp(-abs(x_)/theta))/(2*theta^n*factorial(n-1));
                if asym
                    W(1:offset) = W(1:offset)/4;
                end
                plot(x_, W); hold on;
                %bar(x_, W);
            end
        end
    end

    % Boundary conditions
    for i=1:nStr
        LI(:,i)=circshift(W,offset+i,2); 
    end
    
    % Weight Normalization to maxIN
    LIS=sum(LI');
    for j=1:nStr
        if (LIS)
            LI(j,:) = maxIN * LI(j,:) / LIS(j);
        end
    end
end

if (~bidir)
    for m=1:nStr
        for n=1:nStr
            if (LI(m,n)<LI(n,m) )
                LI(m,n)=0;
            end;
        end;
    end;
end


% Set up feed forward cortical-(thalamo)-cortical connections
inpFF=zeros(nAreas,1);
inpFB=zeros(nAreas,1);
for m=1:nAreas
    inpFF(Ar(m))=m; % Ar(m) : area sending FB signal to m, thus m sends FF signal to Ar(m) 
    CxFF{Ar(m)} = lognrnd(1,1,Cxsz(Ar(m)),Cxsz(m));
    inpFB(m)=Ar(m);
    CxFB{m} = lognrnd(1,1,Cxsz(m),Cxsz(Ar(m)));
end

% Set up corticostriatal inputs
% the pools are the Cx area + its 2 neighboors (previous and next in SFR)
Cx2StrPools = cell(nFrontal,1);
for m=1:nFrontal
    Cx2StrPools{m} = [bnd(Cx2StrAreas(m))+1:bnd(Cx2StrAreas(m)+1) bnd(inpFF(Cx2StrAreas(m)))+1:bnd(inpFF(Cx2StrAreas(m))+1) bnd(inpFB(Cx2StrAreas(m)))+1:bnd(inpFB(Cx2StrAreas(m))+1)];
end
%this loop selects neurons randomly from the pool of the affiliated Cx
%and connects them to the corresponding Str area neurons
Cx2Str = zeros(nStr, nCx2Str);
for m=1:nFrontal
    for n=1:Strsz(m)
        [rn ri] = sort(rand(size(Cx2StrPools{m},2),1));
        Cx2Str(n+Stroffs(m),:) = Cx2StrPools{m}(ri(1:nCx2Str));
    end
end

% Set up dopamine to Str modulation connectiivity
DA2CcStr = zeros(nStr, nCx2Str);
for m=1:nStr
    [rn ri] = sort(rand(nDA,1));
    DA2CcStr(m,:) = ri(1:nCx2Str);
end

% Set up Cortico striatal weights
CxStrW = zeros(nStr, nCx2Str)+CxStrW0;
CxStrWb = cell(1,backups);
for m=1:backups
    CxStrWb{m} = zeros(nStr, nCx);
end
for m=1:nStr
    CxStrWb{1}(m,Cx2Str(m,:))=CxStrW(m,:);
end
CxStrA = ones(nStr, nCx2Str); % CxStr adjacency matrix
dPrePostCxStr = zeros(nStr, nCx2Str);
dPostPreCxStr = zeros(nStr, nCx2Str);
dPrePostCxStrM = zeros(sz, nStr, nCx2Str);
dPostPreCxStrM = zeros(sz, nStr, nCx2Str);
CxStrDAup = zeros(nStr, nCx2Str);
CxStrDAdown= zeros(nStr, nCx2Str);

% Set up direct and indirect pathways
[rn D1] = sort(rand(nStr,1));
D2 = sort(D1(floor(nStr/2)+1:nStr));
D1 = sort(D1(1:floor(nStr/2)));

% Set up Striato-Thalamic connections : the gate G
StrTh = cell(nFrontal, 1);
if ~topographic_StrTh
    for m=1:nFrontal
        StrTh{m}=zeros(Cxsz(m), nStr);
        for n=1:Cxsz(m)
            [rn ri] = sort(rand(Strsz(m),1));  % rn : random number ;  ri : random index
            StrTh{m}(n, ri(1:nStr2Th)+Stroffs(m)) = 1;
        end
        if D2_only
            StrTh{m} = -1 * StrTh{m};
        else
            StrTh{m}(:, D2) = -1 * StrTh{m}(:, D2);
        end
    end
else
    D2 = 1:nStr;
    D1 = D2(1:2:end); % D1 indices = odd numbers
    D2 = D2(2:2:end); % D2 indices = even numbers
    for m=1:nFrontal
        StrTh{m}=zeros(Cxsz(m), round(nStr/2));
        topoMap = zeros(Cxsz(m),1);
        surround = topoMap; 
        center = topoMap;
        surround(1:D2_surround) = -1*ones(D2_surround,1);
        center(D2_surround+1:D2_surround+D1_center) = ones(D1_center,1);
        surround(D2_surround+D1_center+1:D1_center+2*D2_surround) = -1*ones(D2_surround,1);
        
        topoMap = circshift(topoMap,round(-(2*D2_surround+D1_center)/2),1);
        surround = circshift(surround,round(-(2*D2_surround+D1_center)/2),1);
        center = circshift(center,round(-(2*D2_surround+D1_center)/2),1);
        
        szShift = Cxsz(m)/(nStr/4);
        
        for n = 1:floor(nStr/4)
            StrTh{m}(:,D1(n)) = center;  % m*((nStr/2)-1)+n
            StrTh{m}(:,D2(n)) = surround;
            if rand<(szShift-floor(szShift))
                center = circshift(center,ceil(szShift),1);
                surround = circshift(surround,ceil(szShift),1);
            else
                center = circshift(center,floor(szShift),1);
                surround = circshift(surround,floor(szShift),1);
            end
        end
         
        if (m==1) 
            StrTh{m} = [StrTh{m}, zeros(size(StrTh{m}))] ;
        elseif (m==2) 
            StrTh{m} = [zeros(size(StrTh{m})), StrTh{m}] ;
        end;
        figure; imagesc(StrTh{m});
    end
end

%set up tone through receptive field in A1 (center is 12% of field, surround 8%, and neutral 80%)
freq_offset = 37; %where receptive field starts (here we went tone at center of field)
tone = ones(Cxsz(3),1)*0.5; %%baseline (neutral)
center_surround=zeros(19,1); center_surround(5:15)=ones(11,1);
tone(freq_offset:freq_offset+size(center_surround,1)-1)=center_surround;
%tone(offset+1:offset*2)=ones(offset,1);

tone_itv=5000;
tone_dur=1000;

%% Main Loop
idx=1;
for iter=start+1:niters
    idx=idx+1;
    if(mod(iter,10000) == 0) disp(['iter = ', num2str(iter)]); end;

    %% Infomax Learning and Output
    for (m=1:nAreas)
        xxIdx = mod(iter,tauX)+1;
        xx{m}(:,xxIdx) = CxFF{m}*xp(bnd(inpFF(m))+1:bnd(inpFF(m)+1));
        if (m<=nFrontal)
            xx{m}(:,xxIdx) = xx{m}(:,xxIdx).*(2.0*heaviside(StrTh{m}*hx'));
        end
        %introduce auditory input
%         if (m==3)
%             if (iter>=300000 && iter<=800000 && mod(iter,tone_itv)<tone_dur)
%                 xx{m}(:,xxIdx) = xx{m}(:,xxIdx).*tone;
%                 %%xx{m}(:,xxIdx) = xx{m}(:,xxIdx) + tone*mean(bias{inpFF(m)})/tauX; %tone is taken of strength 1/3 of mean input
%             end
%         end
        xx{m}(:,xxIdx) = xx{m}(:,xxIdx) + CxFB{m}*xp(bnd(inpFB(m))+1:bnd(inpFB(m)+1));
        x = sum(xx{m}')';
        
        %compute bias to compute zero mean
        if iter<Xsize
            bias{m} = bias{m} + x/Xsize;
        end
        %compute covariance matrix and whitening matrix
        if (iter>=Xsize)
            q{m}=q{m}+(x-bias{m})*(x'-bias{m}'); %compute covariance once the bias is partially established
            %update whitening matrix based on covariance at interval Xsize
            if (iter>=2*Xsize-1 && mod(iter,Xsize)==Xsize-1)
                q{m}=q{m}/Xsize;
                c{m}=q{m}^(-1/2);
            end
        end
        %turn on ACh at time t_Ach
        if iter==t_Ach
           Ach=Ach_2;
        end
        
        %compute infomax input and update IMAX weight matrix
        if iter>=Xsize
            bias{m} = bias{m} + (x-bias{m}) / Xsize;
            x = x - bias{m};
%             if iter>=2*Xsize
%                 x = c{m}*x;
%             end
            x_s(iter,Cxoffs(m)+1:Cxoffs(m+1)) = x;
            u = w{m}*x;
            u_s(iter,Cxoffs(m)+1:Cxoffs(m+1)) = u;
            yy = 1./(1 + exp(-(u+w0{m})));
            y_s(iter,Cxoffs(m)+1:Cxoffs(m+1)) = yy;
            Q{m} = Q{m} + yy*yy';
            TERM1=1-2*yy;                    
            delta_w{m} = delta_w{m} + (TERM1)*x';                       % accumulate partial learning vector
            delta_w0{m} = delta_w0{m} + (TERM1);                        % accumulte bias learning vector
            sg(idx,bnd(m)+1:bnd(m+1)) = 1.0 - (Ach*(1.0-yy));           % populate the feed forward supragranular (Infomax) input

            if ( mod(iter,invt) == 0) 
                w{m} = w{m} + learning_rate * (invt * pinv(w{m}') + delta_w{m}); %pseudo inverse function computes anti-redundancy term
                w0{m} = w0{m} + learning_rate * delta_w0{m};
                delta_w{m} = zeros(usize(m));
                delta_w0{m} = zeros(usize(m), 1);
            end

            if ( mod(iter,checkpoint) == 0)
                wb{m, backup} = w{m};
                cb{m, backup} = c{m};
            end
            %from t_Ach onwards, set inverted SG signal from previous sim
            %if (iter>=t_Ach)
            %    sg(idx,bnd(m)+1:bnd(m+1)) = inverted_infomax(iter-599999,bnd(m)+1:bnd(m+1));
            %end
        end
        
        %this loop could be put inside the loop above the loop above (line
        %258 at the time of this comment)
        if (iter<Xsize )
            sg(idx,:) = abs(randn(1,NE));
            sg(idx,:) = 1.0 - (Ach * (sg(idx,:)/max(sg(idx,:))));
        end
        
        if (m<=nFrontal)
            sg(idx, bnd(m)+1:bnd(m+1)) = heaviside(x'); %heaviside(StrTh{m}*hx')';
        end
    end
    
    %% Synfire Ring Dynamics
    SWEE=sum(WEEp')';
    %normalize Exc-Exc weights
    for n=1:NE;
        WEEp(:,n)=WEEp(:,n)./SWEE;
    end
    
    xprev=xp;
    xp = heaviside( (WEEp*xprev).*(2.0/(2.0-Ach)*sg(idx,:))' - WEIp*y - TEp );
    SFRE_raster(idx,:) = xp';
    TEp = TEp + eta_IP*(xp-HIPp);
    C = xp*xprev';
    if (iter>niters-10000)
        Csp = Csp + xp*xp';  % as xp is a binary matrix, xp*xp' gives the correlation matrix at current time step
    end
    WEEp = WEEp + eta_STDP * AEEp.*(C-C');
    AEEp = WEEp>0;
    WEEp = WEEp.*AEEp;
    if (rand<0.2)
        avail = find(~AEEp .* A_0 .* ~eye(NE));
        if (~isempty(avail))
            gen = avail(ceil(rand*size(avail,1)));
            WEEp(gen) = 0.001;
            AEEp(gen) = 1;
        end
    end
    
    WEIp = WEIp - eta_inhib * AEIp.*((1-xp*eta_iSTDP)*y'); 
    WEIp = WEIp + AEIp.*(WEIp<0.001).*0.001;
    y = heaviside(WIEp*xprev - TI);% + sigma2_chi*randn(NI,1);
    SFRI_raster(idx,:) = y';

    %% FitzHugh-Nagumo dynamics

    % Striatum local inputs
    g = xfn(idx-1,:).*(xfn(idx-1,:)>0);  
    IN(:,idx) = LI*g'; % LI : str local connectivity matrix    
    % Random input
    if ~mod(iter,Pr)
        % P = Ir*rand(1,nStr); % uniform distribution of random time-varying input
        P = normrnd(Ir, sigmaP, [1,nStr]);
    end
    % Spatially clustered Stimulus
    if (~mod(iter,stim_intv) && iter>=stim_start && iter<stim_end)
        if stim_noise
            B=zeros(npatts, nStr);
            if npatts==4
                B(1,1:25) = normrnd(stim_max, 0.05,[1,25]);
                B(2,26:50) = normrnd(stim_max, 0.05,[1,25]);
                B(3,51:75) = normrnd(stim_max, 0.05,[1,25]);
                B(4,76:100) = normrnd(stim_max, 0.05,[1,25]);
            elseif npatts==6
                B(1,1:16) = normrnd(stim_max, 0.05,[1,16]);
                B(2,17:33) = normrnd(stim_max, 0.05,[1,17]);
                B(3,34:50) = normrnd(stim_max, 0.05,[1,17]);
                B(4,51:66) = normrnd(stim_max, 0.05,[1,16]);
                B(5,67:83) = normrnd(stim_max, 0.05,[1,17]);
                B(6,84:100) = normrnd(stim_max, 0.05,[1,17]);
            elseif npatts==10
                B(1,1:10) = normrnd(stim_max, 0.05,[1,10]);
                B(2,11:20) = normrnd(stim_max, 0.05,[1,10]);
                B(3,21:30) = normrnd(stim_max, 0.05,[1,10]);
                B(4,31:40) = normrnd(stim_max, 0.05,[1,10]);
                B(5,41:50) = normrnd(stim_max, 0.05,[1,10]);
                B(6,51:60) = normrnd(stim_max, 0.05,[1,10]);
                B(7,61:70) = normrnd(stim_max, 0.05,[1,10]);
                B(8,71:80) = normrnd(stim_max, 0.05,[1,10]);
                B(9,81:90) = normrnd(stim_max, 0.05,[1,10]);
                B(10,91:100) = normrnd(stim_max, 0.05,[1,10]);
            end
        end
        S=B(mod(round(iter/stim_intv),npatts)+1,:);
    
    % Post-stimulus consists of transient (& spatially homogeneous) stimulus at sparser interval
    elseif (iter>=stim_end)
        if(mod(iter,stim_intv)==0)
            S = normrnd(stim_max, 0.05, [1,nStr]);
        elseif (mod(iter,stim_intv)==stim_dur)
            %S = zeros(1,nStr);
            S = normrnd(stim_max, 0.05, [1,nStr]);
        end
    end
    % Striatum winnerless update
    for j=1:nStr
        xfn(idx,j) = xfn(idx-1,j) + step1 * ( xfn(idx-1,j) - xfn(idx-1,j)^3/3 - yfn(idx-1,j) - zfn(idx-1,j) * (xfn(idx-1,j) - NU) + Ic + P(1,j) + S(1,j) + CxStr*sum(CxStrW(j,:)'.*xp(Cx2Str(j,:))));
        yfn(idx,j) = yfn(idx-1,j) + step2 * ( xfn(idx-1,j) + afn - bfn * yfn(idx-1,j) );    
        zfn(idx,j) = zfn(idx-1,j) + step3 * ( IN(j,idx) -  zfn(idx-1,j) );
    end;
    hx = heaviside(xfn(idx,:)).*xfn(idx,:);
    hx_s(iter,:) = hx;

	% Corticostriatal learning
    hxusp = (hx-hxprev)>0;
    hxdsp = (hx-hxprev)<0;
    
    CCxStrPrePost = (xp*hxusp)';
    CCxStrPostPre = -(xp*hxdsp)';
    DAup = sum(DAsp)>0;  % DAsp : window dopamine stays in synapse, here summed along the time span
    DAdown = ~DAup;
    GABAupD1 = intersect(find(zfn(idx,:)>GABAtD1),D1);
    GABAdownD1 = intersect(find(zfn(idx,:)<=GABAtD1),D1);
    GABAupD2 = intersect(find(zfn(idx,:)>GABAtD2),D2);
    GABAdownD2 = intersect(find(zfn(idx,:)<=GABAtD2),D2);
    
    for m=1:nStr
        dPrePostCxStr(m,:) = CCxStrPrePost(m,Cx2Str(m,:));
        dPostPreCxStr(m,:) = CCxStrPostPre(m,Cx2Str(m,:));
        CxStrDAup(m,:) = DAup(DA2CcStr(m,:));
        CxStrDAdown(m,:) = DAdown(DA2CcStr(m,:));
    end        

    dPrePostCxStr(GABAupD1,:) = -dPrePostCxStr(GABAupD1,:) .* CxStrDAup(GABAupD1,:);
    dPostPreCxStr(GABAupD1,:) = -dPostPreCxStr(GABAupD1,:) .* CxStrDAup(GABAupD1,:);

    dPrePostCxStr(GABAdownD1,:) = dPrePostCxStr(GABAdownD1,:) .* CxStrDAup(GABAdownD1,:);
    dPostPreCxStr(GABAdownD1,:) = dPostPreCxStr(GABAdownD1,:) .* CxStrDAup(GABAdownD1,:);
    
    dPrePostCxStr(GABAupD2,:) = -dPrePostCxStr(GABAupD2,:) .* CxStrDAup(GABAupD2,:);
    dPostPreCxStr(GABAupD2,:) = -dPostPreCxStr(GABAupD2,:) .* CxStrDAdown(GABAupD2,:);
    
    dPrePostCxStr(GABAdownD2,:) = dPrePostCxStr(GABAdownD2,:) .* CxStrDAdown(GABAdownD2,:);
    dPostPreCxStr(GABAdownD2,:) = dPostPreCxStr(GABAdownD2,:) .* CxStrDAup(GABAdownD2,:);
    
    CxStrW = CxStrW + eta_CxStr*CxStrA.*(dPrePostCxStr + dPostPreCxStr);
    dPrePostCxStrM(idx,:,:) = dPrePostCxStr;
    dPostPreCxStrM(idx,:,:) = dPostPreCxStr;
    CxStrA = CxStrW>0;  
    CxStrW = CxStrW.*CxStrA;
    
    %create new cortico-striatal synapse of strength 0.001 with 20% probability
    for m=1:nFrontal
        [rA0,cA0]=find(CxStrA(Stroffs(m)+1:Stroffs(m+1),:)==0);
        for n=1:size(rA0,1)
            if (rand<0.2)
                gen=setdiff(Cx2StrPools{m},Cx2Str(Stroffs(m)+rA0(n),:));
                [rn,ri]=sort(rand(size(gen,2),1));
                Cx2Str(Stroffs(m)+rA0(n),cA0(n)) = gen(ri(1));
                CxStrW(Stroffs(m)+rA0(n),cA0(n)) = 0.001;
                CxStrA(Stroffs(m)+rA0(n),cA0(n)) = 1;
            end
        end
    end
    
    %normalize synpatic weights
    SCxStrW=sum(CxStrW')';
    for m=1:nCx2Str;
        CxStrW(:,m)=(CxStrW(:,m)./SCxStrW)*0.1;
    end
    
    
    if ( mod(iter,checkpoint) == 0)
        %make backup of Str synaptic weights
        for m=1:nStr
            CxStrWb{backup}(m,Cx2Str(m,:))=CxStrW(m,:);
        end
        %make backup of SFR exc-exc synaptic weights
        weeb(backup,:,:)=WEEp;
        
        backup=backup+1;
    end    
    hxprev = hx;
    
    
    %% Mihalas-Niebur dynamics
    for n=1:nDA
        I_e(idx,n) = I_StrDA*sum(hx(fid(n,:)));
        dI = -k.*I(idx-1,:,n)*dt;
        dV = (1/Cap*(I_e(idx,n)+sum(I(idx-1,:,n))-G*(V(idx-1,n)-E_L)))*dt;
        dTheta = (a*(V(idx-1,n)-E_L)-b*(Theta(idx-1,n)-Theta_inf))*dt;
        ip=1;
        
        I_p(ip,:,n) = I(idx-1,:,n) + dI;        
        V_p(ip,n) = V(idx-1,n) + dV;
        Theta_p(ip,n) = Theta(idx-1,n) + dTheta;

        for p=1:np
            ip=mod(p,2)+1;
            ip_prime=mod(p+1,2)+1;

            I_p(ip,:,n) = I(idx-1,:,n) + 0.5*(dI - k.*I_p(ip_prime,:,n)*dt);      
            V_p(ip,n) = V(idx-1,n) + 0.5*(dV + (1/Cap*(I_e(idx,n)+sum(I_p(ip_prime,:,n))-G*(V_p(ip_prime,n)-E_L)))*dt);
            Theta_p(ip,n) = Theta(idx-1,n) + 0.5*(dTheta + (a*(V_p(ip_prime,n)-E_L)-b*(Theta_p(ip_prime,n)-Theta_inf))*dt);
        end
            
        I(idx,:,n) = I_p(ip,:,n)';
        V(idx,n) = V_p(ip,n);
        Theta(idx,n) = Theta_p(ip,n);
    end
    
    DAidx = mod(iter, tauDA)+1; 
    DAsp(DAidx,:)=V(idx,:)>=Theta(idx,:);
    firing = find(DAsp(DAidx,:));    
    if (~isempty(firing))
        for j=1:nI
           I(idx,j,firing) = R(j)*I(idx,j,firing)+An(j);
        end;
        V(idx,firing) = V_r;
        DA_raster(idx,firing)=1;
        Theta(idx,firing) = max(Theta_r, Theta(idx,firing));
    end
    t=t+dt;
end;
%save('x_s',x_s,'-v7.3');
%save('y_s',y_s,'-v7.3');
%save('u_s',u_s,'-v7.3');
%save('SFRE_raster',SFRE_raster,'-v7.3');
file_name=sprintf('Koralek_v09_noCxStr_etaIP3_etaSTDP2_HIP100Hz_Ir0025_Stim_npatts%i_invt0%i_tauX%i_Ach_%i_gammaKernel%i_bidir%i_nStrTh%i_topoMap%i_%ikts',npatts,invt,tauX,Ach_2*100,gamma_kernel,bidir,nStr2Th,topographic_StrTh,niters/1000);
save(file_name, '-v7.3');