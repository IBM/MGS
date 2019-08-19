#!/usr/bin/env python
import sys, io, os, numpy as np
from scipy.stats import skew
import matplotlib
import matplotlib.pyplot as plt

HELPSTRING="GEN_DAT()\n\n\
NAME\n\tgen_dat -- Computes mean interspike interval information from spike data. Output created where command is executed.\n\n\
SYNOPSIS\n\tgen_dat [-f] [path/to/inputfile] ([-o] [outputfile] [...])\n\n\
DESCRIPTION\n\tThe data should be in the following format:\n\t\tfirst column - spike time, second column - cell index number.\n\n\tThe following options are available:\n\
\n\t-c\tcoefficient of variations distribution ['cv':array[float], 'cell':int]\n\
\n\t-d\tdistribution of mean ISIs across cells ['misi':float, 'cell':int]\n\
\n\t-e\tISI standard error [float]\n\
\n\t-f\tSpecify the input data file to be used.\n\
\n\t-h\tDiplay the help message for gen_dat.\n\
\n\t-o\t(Optional) Specify the output filename to be used.\n\
\n\t-p\tprobability of a value in a CV2 distribution ['upper-range':int, 'prob':int]\n\
\n\t-s\tISI standard deviation [float]\n\
\n\t-w\twipe out old files \n\
"

#formats:
DAT=(float,int)
DST=(np.ndarray,int)
SIM=[('simulation',int)]
SIMSPLIT="\n\n" #Delimiter separating simulations
GESPLIT="\n\n\n" #Delimiter separating set of simulations for each Ge


def parse():
    """
    data,simulations,filename,flags=parse()

    data = array of ndarray of tuple (spike_time, cell_index)
    simulations = array of ndarray of tuple (cell_index, )
    filename = data filename
    flags = a list of arguments

    """
    args = ' '.join(sys.argv[1:]).split()
    infile=np.empty((0,), dtype={'names':('spike','cell'), 'formats':DAT})
    simulations=np.empty((0,), dtype=SIM)
    ge_files=[] #np.empty((0,))
    ge_sims=[] #np.empty((0,))
    outfile=".dat"
    flags=[]

    # track start/end time-point
    time_window=[]

    if '-h' in args:
        #print HELPSTRING
        print(HELPSTRING)
        sys.exit(0)

    if args:
        try:
            while args:
                arg=args.pop(0)
                if arg == '-f':
                    read=0
                    arg=args.pop(0)

                    # open file for time window 
                    path = os.path.dirname(arg)
                    timewindow_file = os.path.join(path, "timewindow_Output.dat")
                    time_window = io.open(timewindow_file, encoding="utf-8").read().split()
                    print('before: ', time_window)
                    print('type: ', type(time_window))
                    time_window = [float(i)/1000 for i in time_window] #convert from string to float 
                    print(time_window)
                    #data=io.open(arg, encoding="utf-8").read().split(SIMSPLIT)
                    sims_by_ge=io.open(arg, encoding="utf-8").read().split(GESPLIT)
                    
                    for ge in range(len(sims_by_ge)):
                        print(ge)
                        data=sims_by_ge[ge].split(SIMSPLIT)
                        
                        for sim in range(len(data)):
                            if data[sim] == "" or data[sim] == "\n":
                                continue
                            file=np.genfromtxt(io.StringIO(data[sim]), dtype={'names':('spike','cell'), 'formats':DAT})
                            infile=np.append(infile, file)
                            simulations=np.append(simulations, np.array([(sim,)]*file.size, dtype=SIM))
                        
                        ge_files.append(infile)
                        ge_sims.append(simulations)
                        infile=np.empty((0,),dtype={'names':('spike','cell'),'formats':DAT})
                        simulations=np.empty((0,),dtype=SIM)
                        print('num_gi_sims: ', len(data))
                    print('len_ge_sims:',len(ge_files))
                elif arg == '-o':
                    read=1
                    outfile=args.pop(0)
                else:
                    flags.append(arg)
        except:
            #print "Missing flag or argument.\n\n"+HELPSTRING
            print("Missing flag or argument.\n\n"+HELPSTRING)
            sys.exit(1)
    else:
        #print HELPSTRING
        print(HELPSTRING)
        sys.exit(1)

    #time_window=time_window/1000 #convert from ms to secs
    return ge_files, ge_sims, outfile, flags, time_window

def clean(path):
    assert (path != os.path.abspath(os.sep))
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        os.rmdir(path)

def binISI(data, col, cells, bins, time_window):
    """calculates mean ISI across all cells for each time window 
    return: array[float]
    """

    """
    Computes mean ISI for sliding window; collapses network into 1 cell for each 
    time window by combining all spikes and then computing mean ISI
    shift_t: indicates how much to shift sliding window 
    """
    window = (time_window[1] - time_window[0])/bins
    shift_t = 0.01 
    meanWins = np.empty((0,))
    for i in range(int(time_window[1]/shift_t)): #range(bins):
        spike_win=data[np.logical_and(data['spike'] < (i*shift_t)+window, data['spike'] >= i*shift_t)]
        print('spike_win: ',spike_win)
        print('len_spike_win: ',len(spike_win))
        if len(spike_win) != 0:
            print('last: ',spike_win['spike'][-1])
            print('first: ',spike_win['spike'][0])
            misi = (spike_win['spike'][-1] - spike_win['spike'][0])/(spike_win.shape[0] - 1)
            meanWins=np.append(meanWins,misi)
        else:
            meanWins=np.append(meanWins,window/2)

    """
    Computes mean ISI for specified number of time windows (i.e. number of 'bins')
    """
    #for i in range(bins):
        #meanWin = np.empty((0,))

        #for cell in cells:
        #    spike=data[col == cell]
        #    spike_win=spike[np.logical_and(spike['spike'] < (i+1)*window, spike['spike'] >= i*window)]
        #    misi = mISI(spike_win,spike_win['cell'],cell)
            #print('std. dev: ', stdISI(spike_win,spike_win['cell'],cell))
            #print('misi: ', misi)
       #     if misi is not None:
       #         meanWin = np.append(meanWin,misi['misi'])
       #     else:
       #         """if no spikes occur in time window, then
       #         mISI = window/2 which will be high in comparison to other 
       #         mISIs to show that little to no spikes occur in this 
       #         time window
       #         """
       #         meanWin = np.append(meanWin,window/2) 
        #print('meanWin: ', meanWin)
       #  if meanWin is not None:
       #      meanWins = np.append(meanWins,np.mean(meanWin))
        
    return meanWins

def binCV(data,col, cells, bins, time_window):
    """ computes mean CV2 across all cells for each time window;
    time window is specified by passing number of bins (i.e. number of windows desired within simulation) 
    return: array[float]
    """
    window = (time_window[1] - time_window[0])/bins
    meanCVs = np.empty((0,))
        
    for i in range(bins):
        meanCV = np.empty((0,))
        for cell in cells:
            spike=data[col == cell]
            spike_win=spike[np.logical_and(spike['spike'] <((i+1)*window), spike['spike'] >= i*window)]
            isi = ISI(spike_win,spike_win['cell'],np.unique(spike_win['cell']))
            cv  = CV2(isi,isi['cell'],np.unique(isi['cell']))
    
            if len(cv) != 0:
                meanCV = np.append(meanCV,np.mean(cv['cv']))
        if meanCV != []:
            meanCVs = np.append(meanCVs,np.mean(meanCV))
    return meanCVs

def ISIbyTime(data, col, cells):
    """mean ISI of a cell for each given time frame
    return: array[]
    """
    mISIs_windows=np.empty((0,), dtype={'names':('misi','cell'),'formats':DST})
    #print('last: ', cells[-1])
    for cell in cells:
        misi=np.empty((0,))
        data_cell=data[col == cell]
        
        for i in range(4):
            #print(data[data['spike'] <= 500])
            inputs = data_cell[np.logical_and(data_cell['spike'] <= (2000/4)*(i+1), data_cell['spike'] >= i*(2000/4))]
            input_col = inputs['cell']
            #input_cells = np.unique(col)
            mean_by_win = mISI(inputs,input_col,cell)
            if mean_by_win is not None:
                misi = np.append(misi,mean_by_win['misi'])
            else:
                np.append(misi,-99999)

        if misi is not None:
            mISIs_windows=np.append(mISIs_windows,np.array((misi,cell),dtype={'names':('misi','cell'),'formats':DST}))

    return mISIs_windows

def stdISIWs(data, col, cells):
    """std ISI of mISIs (across windows) for each cell
    return: array[]
    """
    stdISIs=np.empty((0,), dtype={'names':('std','cell'), 'formats':DST})

    for cell in cells:
        data_cell=data[col == cell]
        std=stdISI(data_cell,col,cell)

        if std is not None:
            stdISIs=np.append(stdISIs,np.array((std,cell),dtype={'names':('std','cell'),'formats':DST}))

    return stdISIs

def reduceCells(data, col, cells, num_cells,threshold=4):
    """reduces number of cells in the network
    return: array[]
    """
    reduce_cells=np.arange(num_cells)
    mean_period_first=np.empty((0,),dtype={'names':('misi','cell'),'formats':DST})
    mean_period_last=np.empty((0,))
    for cell in cells:
        reduce_cond=data[col == cell]
        
        if len(reduce_cond['misi'][0]) >= threshold:
            mean_period_first=np.append(mean_period_first,np.array((reduce_cond['misi'][0][0],cell),dtype={'names':('misi','cell'),'formats':DST}))
            mean_period_last=np.append(mean_period_last,reduce_cond['misi'][0][-1])
        

    mean_period_first.sort(order='misi')

    return mean_period_first['cell'][0:num_cells]

def mISI(data, col, cell, threshold=2):
    """mean ISI of a cell
    parameters: cell - index of the cell
    return: array['misi':float, 'cell':int] or Null
    """
    spikes=data[col == cell]
    
    if len(spikes) >= threshold:
        first=spikes['spike'][0]
        last=spikes['spike'][-1]
        
        mean=(last - first)/(spikes.shape[0]-1)
        return np.array((mean, cell), dtype={'names':('misi','cell'), 'formats':DAT})
    else:
        return None

def mISIs(data, col, cells):
    """mean ISIs of all cells
    return: array['misi':float, 'cell':int]
    """
    distribution=np.empty((0,), dtype={'names':('misi','cell'), 'formats':DAT})
    for cell in cells:
        misi=mISI(data, col, cell)
        if misi:
            distribution=np.append(distribution, misi)
    return distribution

def stdISI(disi, col, cells):
    """ ISI standard deviation across all cells using mISIs
    return: float
    """
    if not len(disi):
        return 0

    # added to accomodate case when needing to take std. dev. of each cell
    # across windows
    if np.std(disi['misi']).all() == 0:
        return np.std(disi['misi'][0])
    return np.std(disi['misi'])

def steISI(disi, col, cells, std=None):
    """ ISI standard error across all cells using mISIs
    return: float
    """
    N=len(disi)
    if not N:
        return 0
    if not std:
        std=stdISI(disi, col, cells)
    return std/np.sqrt(N)

def ISI(data, col, cells, threshold=3):
    """inter-spike intervals for cell
    return: array['isi':array[float], 'cell':int]
    input: data - secs
    output: isi - secs
    """
    isi=np.empty((0,), dtype={'names':('isi','cell'), 'formats':DST})
    for cell in cells:
        spikes=data[col == cell]
    
        if len(spikes) >= threshold:
            distribution=np.empty((0,))
            for n in range(1, len(spikes)):
                deltaT=np.abs((spikes['spike'][n] - spikes['spike'][n-1])) 
                distribution=np.append(distribution, deltaT)
            isi=np.append(isi,np.array((distribution,cell),dtype={'names':('isi','cell'), 'formats':DST}))
    
    return isi

def CV2(data, col, cells, threshold=3):
    """ coefficient of variations distribution of ISI
    data inputted should be ISI values
    return array['cv':array[float], 'cell':int]
    """
    #print(type(data))
    cv2=np.empty((0,), dtype={'names':('cv','cell'), 'formats':DST})
    for cell in cells:
        isis=data[col == cell]['isi'][0]
        
        if len(isis) >= threshold:
            distribution=np.empty((0,))
            for n in range(1, len(isis)):
                #cv=np.abs((data['spike'][n] - data['spike'][n-1]) /
                #    (data['spike'][n] + data['spike'][n-1]))
                
                cv=np.abs((isis[n]-isis[n-1])/
                          (isis[n] + isis[n-1]))
                distribution=np.append(distribution, cv)

            cv2=np.append(cv2,np.array((distribution,cell), dtype={'names':('cv','cell'), 'formats':DST}))

    #print(cv2[0])
    print(len(cv2))
    print(len(cells))
    # ISI = np.zeros(len(cells))
    # for idx, cell in enumerate(cells):
    #     spikes=data[col == cell]
    #     if len(spikes) >= threshold:
    #         distribution=np.empty((0,))
    #         for n in range(1, len(spikes)):
    #             ISI[idx].append(np.abs((data['spike'][n] - data['spike'][n-1]))

    # mISI = np.zeros(len(cells))
    # for idx, cell in enumerate(cells):
    #     mISI[idx] = np.mean(ISI[idx])
    # SD = np.zeros(len(cells))
    # for idx, cell in enumerate(cells):
    #     SD[idx] = np.sqrt(mean(ISI[idx])

    #             cv=np.abs((data['spike'][n] - data['spike'][n-1]) /
    #                 (data['spike'][n] + data['spike'][n-1]))
    #             distribution=np.append(distribution, cv)
    #         cv2=np.append(cv2,np.array((distribution,cell), dtype={'names':('cv','cell'), 'formats':DST}))
    return cv2

def probability(cv2, size=10):
    """ probability of a value in a CV2 distribution
    return: array['pcv':array[int, int], 'cell':int]
    """
    def compute_range(data):
        """ select a good bin size for the data provided
        return: float
        """
        left,right=np.min(data),np.max(data)
        # span=(right-left)
        span=np.median(data)*2
        return left,right,span

    def compute_size(): #TODO
        return float(size)

    pcv=np.empty((0,), dtype={'names':('pcv','cell'), 'formats':DST})
    for data,cell in cv2:
        count=data.size
        left,right,span=compute_range(data)
        size=compute_size()
        bins=np.zeros(int(size))
        increment=span/(size-1)
        addresses=np.array(data/increment, dtype=int)
        for address in addresses:
            bins[min(address,int(size-1))]+=1
        p=np.vstack((np.arange(size)*increment,bins/count)).T
        pcv=np.append(pcv,np.array((p,cell), dtype={'names':('pcv','cell'), 'formats':DST}))
    return pcv

def FR(data,data_isi,col, cells,time_window):
    """mean firing rate across all cells 
    return: float 
    input: data - seconds
    output: mean(cell_fr) - hertz (Hz)
    """
    fire_rate = None
    cell_fr=np.empty((0,))
    spike_col=data['cell']
    for cell in cells:
        isi=data_isi[col == cell]
        spikes=data[spike_col == cell]
        spike_count = len(spikes)
        fire_rate = spike_count /(time_window[1] - time_window[0])
    
        #fire_rate=(len(isi) + 1)/(spikes['spike'][0]/1000 + np.sum(isi['isi']) + (2000-spikes['spike'][-1])/1000)
        cell_fr=np.append(cell_fr,fire_rate)

    #plt.hist(cell_fr,100)
    #plt.show()
    return np.mean(cell_fr)

def calculate(spikes, flags, time_window, cells=None):
    """mean ISI distribution and standard deviation by cells for a simulation"""
    #print('time_window_calc: ',time_window)
    col=spikes['cell']
    disi=None
    cv2=None
    pcv=None
    std=None
    ste=None
    isi=None
    tisi=None
    std_isis_win=None
    num_bins=100

    if not cells:
        cells=np.unique(col)

    distribution=mISIs(spikes, col, cells)
    if not flags or "-d" in flags:
        disi=distribution
    if not flags or "-s" in flags:
        std=stdISI(distribution, col, cells)
    if not flags or "-e" in flags:
        ste=steISI(distribution, col, cells,)

    tisi=ISIbyTime(spikes,col,cells)
    bisi=binISI(spikes,col,cells,num_bins,time_window)
    std_isis_win=stdISIWs(tisi,tisi['cell'],np.unique(tisi['cell']))
    reduced_net=reduceCells(tisi,tisi['cell'],np.unique(tisi['cell']),10)
    # print(spikes)
    # print("--")
    # print(col)
    # print("--")
    # print(cells)
    # print("--")
    bcv=binCV(spikes,col,cells,num_bins,time_window)
    isi=ISI(spikes,col,cells)
    fr=FR(spikes,isi,isi['cell'],np.unique(isi['cell']),time_window)
    distribution=CV2(isi,isi['cell'],np.unique(isi['cell']))

    if not flags or "-c" in flags:
        cv2=distribution
    if not flags or "-p" in flags:
        pcv=probability(distribution)
    return disi, isi, bisi, bcv, std, ste, cv2, pcv, tisi, std_isis_win, reduced_net, fr

def write(ge_data,simulations,filename,flags,time_window):
    fr_by_ge=list()
    aCV=list() # mean CV 
    aISI=list() # mean ISI
    aSISI=list() # skew ISI
    aRISI=list() # rescaled skew ISI

    for ge in range(len(simulations)):
        sims=np.unique(simulations[ge])
        data=ge_data[ge]

        options = "w+"
        fr = np.empty((0,)) # firing rate for each gi across all cells 
        mCV = np.empty((0,)) # mean CV for each gi for all cells
        meanISI = np.empty((0,)) # mean ISI for each gi for all cells 
        sISI = np.empty((0,)) # skew ISI for each gi for all cells 
        rsISI = np.empty((0,)) #rescaled skew ISI (S/CV) for each gi for all cells 
        
        if flags and "-w" in flags:
            options = "w"
        if not flags or "-d" in flags:
            f_disi=open('misi_'+str(ge+1)+'_'+filename, options)
        if not flags or "-s" in flags:
            f_std=open('std_'+str(ge+1)+'_'+filename, options)
        if not flags or "-e" in flags:
            f_ste=open('ste_'+str(ge+1)+'_'+filename, options)
        if not flags or "-t" in flags:
            f_tisi=open('tisi_'+str(ge+1)+'_'+filename, options)
        if not flags or "-i" in flags:
            f_stdws=open('stdws_'+str(ge+1)+'_'+filename, options)
        if not flags or "-x" in flags:
            f_isiw=open('isiw_'+str(ge+1)+'_'+filename,options)
        if not flags or "-b" in flags:
            f_cvw=open('cvw_'+str(ge+1)+'_'+filename,options)

        f_rnet=open('rnet'+str(ge+1)+'_'+filename, options)

        cdir=os.path.join('cv2/ge_' + str(ge+1) + '/')
        if os.path.exists(cdir):
            clean(cdir)
        if not flags or "-c" in flags:
            os.makedirs(cdir)

            pdir=os.path.join('pcv/ge_' + str(ge+1) + '/')
        if os.path.exists(pdir):
            clean(pdir)
        if not flags or "-p" in flags:
            os.makedirs(pdir)
        
        for s in range(sims.size):
            simulation=sims[s]
            spikes=data[simulations[ge] == simulation]

            """
            converting the spikes from milliseconds to seconds 
            """
            spikes['spike'] = spikes['spike']/1000

            disi,isi,bisi,bcv,std,ste,cv2,pcv,tisi,std_isis_win,reduced_net,mfr=calculate(spikes, flags,time_window)
        
            fr=np.append(fr,mfr)
        
            #meanISI=np.append(meanISI,mean(isi))
            meanISI=np.append(meanISI,np.mean(disi['misi']))
            mcv = mean(cv2)
            sisi = skewISI(isi)
            mCV=np.append(mCV,mcv)
            sISI=np.append(sISI,sisi)
            rsISI=np.append(rsISI,sisi/mcv)

            if disi is not None:
                for misi,cell in disi:
                    f_disi.write(str(misi)+' '+str(cell)+"\n")
            if not flags or "-d" in flags:
                f_disi.write('\n')
            if std:
                f_std.write(str(std)+"\n")
            if not flags or "-s" in flags:
                f_std.write('\n')
            if ste:
                f_ste.write(str(ste)+"\n")
            if not flags or "-e" in flags:
                f_ste.write('\n')
            if cv2 is not None:
                for cv,cell in cv2:
                    f_cv2=open(os.path.join(cdir, 'cv2_'+str(cell)+filename), "a+")
                    for v in cv:
                        f_cv2.write(str(v)+' '+str(cell)+'\n')
                    f_cv2.write('\n')
                    f_cv2.close()
            if pcv is not None:
                for pb,cell in pcv:
                    f_pcv=open(os.path.join(pdir, 'pcv_'+str(cell)+filename), "a+")
                    for x,y in pb:
                        f_pcv.write(str(x)+' '+str(y)+'\n')
                    f_pcv.write('\n')
                    f_pcv.close()

            if tisi is not None:
                for window,cell in tisi:
                    for misi in window:
                        f_tisi.write(str(misi)+' '+str(cell)+"\n")

            if not flags or "-t" in flags:
                f_tisi.write('\n')

            if std_isis_win is not None:
                for std,cell in std_isis_win:
                    f_stdws.write(str(std)+' '+str(cell)+'\n')

            if not flags or "-i" in flags:
                f_stdws.write('\n')

            if bisi is not None:
                for isi_win in bisi:
                    f_isiw.write(str(isi_win)+'\n')

            if not flags or "-x" in flags:
                f_isiw.write('\n')

            if bcv is not None:
                for cv_win in bcv:
                    f_cvw.write(str(cv_win)+'\n')

            if not flags or "-b" in flags:
                f_cvw.write('\n')

            #keep_cells=np.isin(spikes['cell'],reduced_net)
            #for spike,cell in spikes[keep_cells]:
            #    f_rnet.write(str(spike)+' '+str(cell)+'\n')
            f_rnet.write(str(reduced_net))
            f_rnet.write('\n')

        # Plot mean firing for varying inhibition I (g_i)
        #if(len(fr) > 1):
            #plt.loglog(np.arange(0,65,5),fr)
            #plt.xlabel('inhibition I')
            #plt.ylabel('firing rate (Hz)')
            #plt.title('Firing Rate for Ge = 25')
            #plt.show()

        f_rnet.close()

        if not flags or "-d" in flags:
            f_disi.close()
        if not flags or "-s" in flags:
            f_std.close()
        if not flags or "-e" in flags:
            f_ste.close()
        if not flags or "-t" in flags:
            f_tisi.close()
        if not flags or "-i" in flags:
            f_stdws.close()
        if not flags or "-x" in flags:
            f_isiw.close()
    
        fr_by_ge.append(fr)
        aCV.append(mCV)
        aISI.append(meanISI)
        aSISI.append(sISI)
        aRISI.append(rsISI)

    plot_legend=['E=15', 'E=25', 'E=40']
    plotFR(fr_by_ge,plot_legend)
    plotmISICV(aCV,plot_legend)
    #plotSTATs(['mean ISI','ISI CV','ISI skew'],['WT model', 'HD model'],[2.3, 3.4, 5], [7.5, 4.1, 1.2])
    plotSTATs(['mean ISI', 'ISI CV', 'ISI skew',  'ISI res. skew', 'firing rate'], plot_legend, [aISI[0][3], aCV[0][3], aSISI[0][3], aRISI[0][3], fr_by_ge[0][3]], [aISI[1][3], aCV[1][3], aSISI[1][3], aRISI[1][3], fr_by_ge[1][3]], [aISI[2][3], aCV[2][3], aSISI[2][3], aRISI[2][3], fr_by_ge[2][3]])

def mean(data):
    """computes mean of data across cells for a given gi value 
    return: float 
    """
    mean=np.empty((0,))
    for value,cell in data:
        mean = np.append(mean,np.mean(value))
    
    return np.mean(mean)

def skewISI(data):
    """computes sample skewness of ISI data across cells for a given gi value
    return: float
    """
    mean = np.empty((0,))
    for isi,cell in data:
        mean = np.append(mean,np.mean(isi))
    
    return skew(mean)

def plotmISICV(CVs,legend):
    """plot mean ISI CV2 across values of excitatory variable Ge 
    return:
    """
    palette = plt.get_cmap('Set1')
    for i in range(len(CVs)-1):
        plt.plot(np.arange(0,35,5),np.asarray(CVs[i]), color=palette(i+1), label=str(legend[i]))
    plt.xlabel('inhibition I')
    plt.ylabel('ISI CV')
    plt.legend(loc=2, ncol=2)
    plt.show()

def plotFR(fire_rates,legend):
    """plot mean firing rate across values of excitatory variable Ge
    return: 
    """
    palette = plt.get_cmap('Set1')
    for i in range(len(fire_rates)-1):
        plt.loglog(np.arange(0,35,5),np.asarray(fire_rates[i]),marker='',color=palette(i+1), label=str(legend[i]))
    plt.xlabel('inhibition I')
    plt.ylabel('firing rate (Hz)')
    plt.legend(loc=2, ncol=2)
    plt.title('Mean Firing Rate across Ge')
    plt.show()

def plotSTATs(labels,legend,*args):
    width = 1/len(labels)
    x = np.arange(len(labels))
    num_label = len(args)
    idx = 0

    fig, ax = plt.subplots()
    for arg in args:
        print('arg: ', arg)
        ax.bar(x+(idx*width), arg, width=width, label=str(legend[idx]))
        idx = idx + 1
    
    ax.set_ylabel('value')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_title('MSN 2500-cell Models')
    plt.show()

def main():
    data,simulations,filename,flags,time_win=parse()
    c=write(data,simulations,filename,flags,time_win)

main()


