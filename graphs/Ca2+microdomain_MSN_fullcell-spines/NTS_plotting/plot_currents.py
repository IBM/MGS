"""
plot currents
"""
# import matplotlib.pyplot as plt
# from unused_codes import *
# import matplotlib as mpl
# matplotlib.use('gtk')
import logging
# import re
# import btmorph
import math

# import numpy as np
import os.path
# import csv
# import fnmatch
from file_utils import write_pdf_new_page, write_pdf_new_page_overlap # NOQA
from file_utils import write_pdf_new_page_currents  # NOQA
from file_utils import write_pdf_new_page_calcium_dye  # NOQA
from file_utils import create_new_pdf_file, get_file, close_pdf_file  # NOQA
from file_utils import get_working_folder, get_pdf_folder  # NOQA
from plot_case9_adv_pdf import plot_case9_adv_pdf  # NOQA
from old_plots import plotcurrent_in_group_multiple_compartments_one_branch
from old_plots import plotcurrent_in_group
from file_utils import get_list_indices  # NOQA
from file_utils import write_pdf_new_page, write_pdf_new_page_overlap_spines # NOQA
# from old_plots import plotcurrent_in_group_spines
# from .unused_codes import *
__author__ = "Hoang Trong Minh Tuan"
__copyright__ = "Copyright 2016, IBM"
__credits__ = []  # list of people who file pugs
__version__ = "0.1"
__maintainer__ = "Hoang Trong Minh Tuan"
__email__ = "tmhoangt@us.ibm.com"
__status__ = "Prototype"  # (Prototype, Development, Production)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s')
# logging.basicConfig(filename='log_plot.txt', level=logging.DEBUG,
# format='%(asctime)s - %(levelname)s - %(message)s')


def plot_msn_microdomain_pdf():
    """
    Plot figures to PDF file
    """
    # more comprehensive deal with multiple processes I/O
    # with spines
    # output directly to PDF file

    ##########
    # USER-DEFINED SECTION
    time_start = 0.0
    # time_start = 19.0
    # time_start = 390.0
    time_end = -1.0
    # time_end = 60.0
    # time_start = 270.0
    # time_end = 380.0
    ########
    global NTS_OUTPUT_PDF

    main_folder, foldername_only = get_working_folder()
    folder = os.path.join(main_folder, foldername_only)
    logging.info("Working on: %s", str(folder))
    NTS_OUTPUT_PDF = get_pdf_folder()

    # page size (3 rows, 2 columns)
    # fig_01 = plt.figure(0)
    # f1, axarr1 = plt.subplots(3, 2)
    # axarr = axarr1

    # common configuration
    # mpl.rcParams['lines.linewidth'] = 2
    # mpl.rcParams['legend.fontsize'] = 13
    # mpl.rcParams['legend.linewidth'] = 2

    # global pdf_file_handlers  # keep the list of all opening pdf files
    pdf_file_handlers = {}

    pdf_filename = NTS_OUTPUT_PDF+"/"+foldername_only+".pdf"
    create_new_pdf_file(pdf_file_handlers, pdf_filename)
    ##########################################################
    # Page 1
    ############################
    my_file = folder+'/'+get_file(folder, 'somaCurrents.dat')
    label_yaxis = "soma"
    plot_type = "channelCurrent"
    width_inch = 10
    height_inch = 3
    write_pdf_new_page(
        pdf_file_handlers,
        my_file,
        pdf_filename,
        time_start,
        time_end,
        label_yaxis,
        plot_type,
        width_inch,
        height_inch)
    ##########################################################
    # Page 1a
    ############################
    my_file = folder+'/'+get_file(folder, 'somaCurrents.dat')
    label_yaxis = "soma"
    plot_type = "channelCurrent"
    width_inch = 10
    height_inch = 3
    show_channels = ["Nat", "KAf", "KAs"]
    write_pdf_new_page(
        pdf_file_handlers,
        my_file,
        pdf_filename,
        time_start,
        time_end,
        label_yaxis,
        plot_type,
        width_inch=width_inch,
        height_inch=height_inch,
        show_channels=show_channels)
    ##########################################################
    # Page 1b
    ############################
    my_file = folder+'/'+get_file(folder, 'somaCurrents.dat')
    label_yaxis = "soma"
    plot_type = "channelCurrent"
    width_inch = 10
    height_inch = 3
    hide_channels = ["Nat", "KAf", "KAs"]
    write_pdf_new_page(
        pdf_file_handlers,
        my_file,
        pdf_filename,
        time_start,
        time_end,
        label_yaxis,
        plot_type,
        width_inch=width_inch,
        height_inch=height_inch,
        hide_channels=hide_channels)
    ##########################################################
    # Page 1c
    ############################
    my_file = folder+'/'+get_file(folder, 'somaCurrents.dat')
    label_yaxis = "soma"
    plot_type = "channelCurrent"
    width_inch = 10
    height_inch = 3
    show_channels = ["NCX"]
    write_pdf_new_page(
        pdf_file_handlers,
        my_file,
        pdf_filename,
        time_start,
        time_end,
        label_yaxis,
        plot_type,
        width_inch=width_inch,
        height_inch=height_inch,
        show_channels=show_channels)
    ##########################################################
    # Page 2
    ############################
    label_yaxis = "soma"
    plot_type = "voltage"
    width_inch = 10
    height_inch = 3
    my_file = folder+'/'+get_file(folder, 'somaV.dat')
    continue_figure = False
    keep_holding_figure = True
    custom_labels = ['model']
    write_pdf_new_page(
        pdf_file_handlers,
        my_file,
        pdf_filename,
        time_start,
        time_end,
        label_yaxis,
        plot_type,
        width_inch,
        height_inch,
        keep_holding_figure=keep_holding_figure,
        continue_figure=continue_figure,
        custom_labels=custom_labels)
    # D1-MSN 200pA
    my_file = get_file("./", 'D1-Cepeda-200pA.txt')
    keep_holding_figure = True
    continue_figure = True
    offset_xaxis = 4.0  # ms
    custom_labels = ['D1-MSN']
    write_pdf_new_page(
        pdf_file_handlers,
        my_file,
        pdf_filename,
        time_start,
        time_end,
        label_yaxis,
        plot_type,
        width_inch,
        height_inch,
        keep_holding_figure=keep_holding_figure,
        continue_figure=continue_figure,
        custom_labels=custom_labels,
        offset_xaxis=offset_xaxis)
    # D2-MSN 200pA
    my_file = get_file("./", 'D2-Cepeda-200pA.txt')
    keep_holding_figure = False
    continue_figure = True
    offset_xaxis = 370.0  # ms
    custom_labels = ['D2-MSN']
    write_pdf_new_page(
        pdf_file_handlers,
        my_file,
        pdf_filename,
        time_start,
        time_end,
        label_yaxis,
        plot_type,
        width_inch,
        height_inch,
        keep_holding_figure=keep_holding_figure,
        continue_figure=continue_figure,
        custom_labels=custom_labels,
        offset_xaxis=offset_xaxis)

    ##########################################################
    # Page 3
    ############################
    plot_type = "voltagecalcium"
    my_file = folder+'/'+get_file(folder, 'somaV.dat')
    my_file2 = folder+'/'+get_file(folder, 'somaCa.dat')
    write_pdf_new_page_overlap(
        pdf_file_handlers,
        my_file,
        my_file2,
        pdf_filename,
        time_start,
        time_end,
        label_yaxis,
        plot_type,
        width_inch,
        height_inch)

    ##########################################################
    # Page 4
    ############################
    my_file = folder+'/'+get_file(folder, 'basalDenV.dat')
    keep_holding_figure = False
    continue_figure = False
    label_yaxis = "basal"
    plot_type = "voltage"
    width_inch = 10
    height_inch = 3
    custom_labels = ['Vm']
    plot_columns = [6]
    write_pdf_new_page(
        pdf_file_handlers,
        my_file,
        pdf_filename,
        time_start,
        time_end,
        label_yaxis,
        plot_type,
        width_inch,
        height_inch,
        keep_holding_figure=keep_holding_figure,
        continue_figure=continue_figure,
        custom_labels=custom_labels,
        plot_columns=plot_columns,
        plot_all=True)
    ##########################################################
    # Page 5
    ############################
    my_file = folder+'/'+get_file(folder, 'basalDenCa.dat')
    keep_holding_figure = False
    continue_figure = False
    label_yaxis = "basal"
    plot_type = "concentrationCalcium"
    width_inch = 10
    height_inch = 3
    custom_labels = ['Calcium']
    plot_columns = [6]
    write_pdf_new_page(
        pdf_file_handlers,
        my_file,
        pdf_filename,
        time_start,
        time_end,
        label_yaxis,
        plot_type,
        width_inch,
        height_inch,
        keep_holding_figure=keep_holding_figure,
        continue_figure=continue_figure,
        custom_labels=custom_labels,
        plot_columns=plot_columns,
        plot_all=True)

    ##########################################################
    # Page 6
    ############################
    my_file = folder+'/'+get_file(folder, 'basalDenCa.dat')
    keep_holding_figure = False
    continue_figure = False
    label_yaxis = "basal"
    plot_type = "concentrationCalcium"
    width_inch = 10
    height_inch = 3
    custom_labels = [r'$\Delta$F/F0']
    plot_columns = [6]
    fluorescence_total = 100.0  # [uM]
    # Kd  = 1.4  # [uM] - Fluo-4 in vivo
    Kd = 0.35  # [uM] - Fluo-4 in vitro [Hagen et al., 2012]
    basal_calcium = 0.1  # [uM]
    write_pdf_new_page_calcium_dye(
        pdf_file_handlers,
        my_file,
        pdf_filename,
        fluorescence_total,
        Kd,
        basal_calcium,
        time_start,
        time_end,
        label_yaxis,
        plot_type,
        width_inch,
        height_inch,
        keep_holding_figure=keep_holding_figure,
        continue_figure=continue_figure,
        custom_labels=custom_labels,
        plot_columns=plot_columns,
        plot_all=True)
    # FINAL
    close_pdf_file(pdf_file_handlers, pdf_filename)
    import shutil
    shutil.copy(pdf_filename, folder)
    return
    # END


def plot_test():
    """
    Plot figures to PDF file
    group plotting together
    """
    # more comprehensive deal with multiple processes I/O
    # with spines
    # output directly to PDF file

    ##########
    # USER-DEFINED SECTION
    time_start = 00.0
    # time_start = 19.0
    # time_start = 390.0
    time_end = -1.0
    # time_end = 60.0
    # time_start = 270.0
    # time_end = 380.0
    ########
    global NTS_OUTPUT_PDF

    main_folder, foldername_only = get_working_folder()
    folder = os.path.join(main_folder, foldername_only)
    logging.info("Working on: %s", str(folder))
    NTS_OUTPUT_PDF = get_pdf_folder()

    # page size (3 rows, 2 columns)
    # fig_01 = plt.figure(0)
    # f1, axarr1 = plt.subplots(3, 2)
    # axarr = axarr1

    # common configuration
    # mpl.rcParams['lines.linewidth'] = 2
    # mpl.rcParams['legend.fontsize'] = 13
    # mpl.rcParams['legend.linewidth'] = 2

    # global pdf_file_handlers  # keep the list of all opening pdf files
    pdf_file_handlers = {}

    pdf_filename = NTS_OUTPUT_PDF+"/"+foldername_only+".pdf"
    create_new_pdf_file(pdf_file_handlers, pdf_filename)

    ##########################################################
    # DENDRITE
    ############################
    my_file = folder+'/'+get_file(folder, 'basalDenCaDomain.dat')
    keep_holding_figure = False
    continue_figure = False
    label_yaxis = "dendrite"
    plot_type = "concentrationCalcium"
    width_inch = 10
    height_inch = 3
    custom_labels = ['Calcium-domain']
    plot_columns = [2]
    write_pdf_new_page(
        pdf_file_handlers,
        my_file,
        pdf_filename,
        time_start,
        time_end,
        label_yaxis,
        plot_type,
        width_inch,
        height_inch,
        keep_holding_figure=keep_holding_figure,
        continue_figure=continue_figure,
        custom_labels=custom_labels,
        plot_columns=plot_columns,
        plot_all=True)
    # FINAL
    close_pdf_file(pdf_file_handlers, pdf_filename)
    import shutil
    shutil.copy(pdf_filename, folder)
    return
    # END


def plot_msn_microdomain_pdf_repeats():
    """
    Plot figures to PDF file
    group plotting together
    """
    # more comprehensive deal with multiple processes I/O
    # with spines
    # output directly to PDF file

    ##########
    # USER-DEFINED SECTION
    time_start = 00.0
    # time_start = 19.0
    # time_start = 390.0
    time_end = -1.0
    # time_end = 60.0
    # time_start = 270.0
    # time_end = 380.0
    ########
    global NTS_OUTPUT_PDF

    main_folder, foldername_only = get_working_folder()
    folder = os.path.join(main_folder, foldername_only)
    logging.info("Working on: %s", str(folder))
    NTS_OUTPUT_PDF = get_pdf_folder()

    # page size (3 rows, 2 columns)
    # fig_01 = plt.figure(0)
    # f1, axarr1 = plt.subplots(3, 2)
    # axarr = axarr1

    # common configuration
    # mpl.rcParams['lines.linewidth'] = 2
    # mpl.rcParams['legend.fontsize'] = 13
    # mpl.rcParams['legend.linewidth'] = 2

    # global pdf_file_handlers  # keep the list of all opening pdf files
    pdf_file_handlers = {}

    pdf_filename = NTS_OUTPUT_PDF+"/"+foldername_only+".pdf"
    create_new_pdf_file(pdf_file_handlers, pdf_filename)

    ##########################################################
    # Page 1+ - plot Istim wave form
    ############################
    label_yaxis = r'$I\_{stim}$'
    plot_type = "currentStim"
    width_inch = 10
    height_inch = 3
    my_file = folder+'/'+get_file(folder, 'Istim.dat')
    continue_figure = False
    keep_holding_figure = False
    custom_labels = [r'$I\_{stim}$']
    write_pdf_new_page(
        pdf_file_handlers,
        my_file,
        pdf_filename,
        time_start,
        time_end,
        label_yaxis,
        plot_type,
        width_inch,
        height_inch,
        keep_holding_figure=keep_holding_figure,
        continue_figure=continue_figure,
        custom_labels=custom_labels)
    ##########################################################
    # Page 3 (temp moving here)
    ############################
    plot_type = "voltagecalcium"
    my_file = folder+'/'+get_file(folder, 'somaV.dat')
    my_file2 = folder+'/'+get_file(folder, 'somaCa.dat')
    label_yaxis = r'soma'
    write_pdf_new_page_overlap(
        pdf_file_handlers,
        my_file,
        my_file2,
        pdf_filename,
        time_start,
        time_end,
        label_yaxis,
        plot_type,
        width_inch,
        height_inch)

    ##########################################################
    # Page 1+ - plot all soma currents
    ############################
    label_yaxis = "soma"
    plotcurrent_in_group_Vclamp(pdf_file_handlers, folder, 'somaCurrents.dat', label_yaxis,
                                pdf_filename,
                                time_start,
                                time_end)
    ##########################################################
    # Page 1+ - plot all soma currents - Only if Vclamp folder (auto-detect)
    ############################
    label_yaxis = "soma"
    plotcurrent_in_group(pdf_file_handlers, folder, 'somaCurrents.dat', label_yaxis,
                         pdf_filename,
                         time_start,
                         time_end)
    ##########################################################
    # Page 1+ - plot all currents in a dendrite
    ############################
    label_yaxis = "basalDen"
    plotcurrent_in_group_multiple_compartments_one_branch(pdf_file_handlers, folder,
                                                          'basalDenCurrents.dat', label_yaxis,
                                                          pdf_filename,
                                                          time_start,
                                                          time_end)
    ##########################################################
    # Page 1c
    ############################
    my_file = folder+'/'+get_file(folder, 'basalDenCurrents.dat')
    # label_yaxis = "soma"
    plot_type = "channelCurrent"
    width_inch = 10
    height_inch = 3
    show_channels = ["CaR", "CaT", "CaN", "CaPQ"]
    write_pdf_new_page_currents(
        pdf_file_handlers,
        my_file,
        pdf_filename,
        time_start,
        time_end,
        label_yaxis,
        plot_type,
        width_inch=width_inch,
        height_inch=height_inch,
        show_channels=show_channels)
    my_file = folder+'/'+get_file(folder, 'basalDenCurrents.dat')
    # label_yaxis = "soma"
    plot_type = "channelCurrent"
    width_inch = 10
    height_inch = 3
    show_channels = ["CaLv12", "CaLv13"]
    write_pdf_new_page_currents(
        pdf_file_handlers,
        my_file,
        pdf_filename,
        time_start,
        time_end,
        label_yaxis,
        plot_type,
        width_inch=width_inch,
        height_inch=height_inch,
        show_channels=show_channels)
    ##########################################################
    # SOMA
    ############################
    label_yaxis = "soma"
    plot_type = "voltage"
    width_inch = 10
    height_inch = 3
    my_file = folder+'/'+get_file(folder, 'somaV.dat')
    continue_figure = False
    keep_holding_figure = True
    custom_labels = ['model']
    write_pdf_new_page(
        pdf_file_handlers,
        my_file,
        pdf_filename,
        time_start,
        time_end,
        label_yaxis,
        plot_type,
        width_inch,
        height_inch,
        keep_holding_figure=keep_holding_figure,
        continue_figure=continue_figure,
        custom_labels=custom_labels)
    # D1-MSN 200pA
    my_file = get_file("./", 'D1-Cepeda-200pA.txt')
    keep_holding_figure = True
    continue_figure = True
    offset_xaxis = 4.0  # ms
    custom_labels = ['D1-MSN']
    write_pdf_new_page(
        pdf_file_handlers,
        my_file,
        pdf_filename,
        time_start,
        time_end,
        label_yaxis,
        plot_type,
        width_inch,
        height_inch,
        keep_holding_figure=keep_holding_figure,
        continue_figure=continue_figure,
        custom_labels=custom_labels,
        offset_xaxis=offset_xaxis)
    # D2-MSN 200pA
    my_file = get_file("./", 'D2-Cepeda-200pA.txt')
    keep_holding_figure = False
    continue_figure = True
    offset_xaxis = 370.0  # ms
    custom_labels = ['D2-MSN']
    write_pdf_new_page(
        pdf_file_handlers,
        my_file,
        pdf_filename,
        time_start,
        time_end,
        label_yaxis,
        plot_type,
        width_inch,
        height_inch,
        keep_holding_figure=keep_holding_figure,
        continue_figure=continue_figure,
        custom_labels=custom_labels,
        offset_xaxis=offset_xaxis)

    ##########################################################
    # Page 3
    ############################
    plot_type = "voltagecalcium"
    my_file = folder+'/'+get_file(folder, 'somaV.dat')
    my_file2 = folder+'/'+get_file(folder, 'somaCa.dat')
    write_pdf_new_page_overlap(
        pdf_file_handlers,
        my_file,
        my_file2,
        pdf_filename,
        time_start,
        time_end,
        label_yaxis,
        plot_type,
        width_inch,
        height_inch)

    ##########################################################
    # AIS
    ############################
    my_file = folder+'/'+get_file(folder, 'AISV.dat')
    keep_holding_figure = False
    continue_figure = False
    label_yaxis = "AIS"
    plot_type = "voltage"
    width_inch = 10
    height_inch = 3
    custom_labels = ['Vm']
    plot_columns = [6]
    write_pdf_new_page(
        pdf_file_handlers,
        my_file,
        pdf_filename,
        time_start,
        time_end,
        label_yaxis,
        plot_type,
        width_inch,
        height_inch,
        keep_holding_figure=keep_holding_figure,
        continue_figure=continue_figure,
        custom_labels=custom_labels,
        plot_columns=plot_columns,
        plot_all=True)
    ##########################################################
    # Page 5
    ############################
    my_file = folder+'/'+get_file(folder, 'AISCaDomain.dat')
    keep_holding_figure = False
    continue_figure = False
    label_yaxis = "AIS"
    plot_type = "concentrationCalcium"
    width_inch = 10
    height_inch = 3
    custom_labels = ['Calcium-domain']
    plot_columns = [6]
    write_pdf_new_page(
        pdf_file_handlers,
        my_file,
        pdf_filename,
        time_start,
        time_end,
        label_yaxis,
        plot_type,
        width_inch,
        height_inch,
        keep_holding_figure=keep_holding_figure,
        continue_figure=continue_figure,
        custom_labels=custom_labels,
        plot_columns=plot_columns,
        plot_all=True)
    ##########################################################
    # Page 5
    ############################
    my_file = folder+'/'+get_file(folder, 'AISCa.dat')
    keep_holding_figure = False
    continue_figure = False
    label_yaxis = "AIS"
    plot_type = "concentrationCalcium"
    width_inch = 10
    height_inch = 3
    custom_labels = ['Calcium']
    plot_columns = [6]
    write_pdf_new_page(
        pdf_file_handlers,
        my_file,
        pdf_filename,
        time_start,
        time_end,
        label_yaxis,
        plot_type,
        width_inch,
        height_inch,
        keep_holding_figure=keep_holding_figure,
        continue_figure=continue_figure,
        custom_labels=custom_labels,
        plot_columns=plot_columns,
        plot_all=True)

    ##########################################################
    # DENDRITE
    ############################
    index_data_dendrite = 2
    my_file = folder+'/'+get_file(folder, 'basalDenV.dat')
    keep_holding_figure = False
    continue_figure = False
    label_yaxis = "basal"
    plot_type = "voltage"
    width_inch = 10
    height_inch = 3
    custom_labels = ['Vm']
    plot_columns = [index_data_dendrite]
    write_pdf_new_page(
        pdf_file_handlers,
        my_file,
        pdf_filename,
        time_start,
        time_end,
        label_yaxis,
        plot_type,
        width_inch,
        height_inch,
        keep_holding_figure=keep_holding_figure,
        continue_figure=continue_figure,
        custom_labels=custom_labels,
        plot_columns=plot_columns,
        plot_all=True)
    ##########################################################
    # Page 5
    ############################
    my_file = folder+'/'+get_file(folder, 'basalDenCaDomain.dat')
    keep_holding_figure = False
    continue_figure = False
    label_yaxis = "dendrite"
    plot_type = "concentrationCalcium"
    width_inch = 10
    height_inch = 3
    custom_labels = ['Calcium-domain']
    plot_columns = [index_data_dendrite]
    write_pdf_new_page(
        pdf_file_handlers,
        my_file,
        pdf_filename,
        time_start,
        time_end,
        label_yaxis,
        plot_type,
        width_inch,
        height_inch,
        keep_holding_figure=keep_holding_figure,
        continue_figure=continue_figure,
        custom_labels=custom_labels,
        # plot_columns=plot_columns,
        plot_all=True)
    ##########################################################
    # Page 5
    ############################
    my_file = folder+'/'+get_file(folder, 'basalDenCa.dat')
    keep_holding_figure = False
    continue_figure = False
    label_yaxis = "basal"
    plot_type = "concentrationCalcium"
    width_inch = 10
    height_inch = 3
    custom_labels = ['Calcium']
    plot_columns = [index_data_dendrite]
    write_pdf_new_page(
        pdf_file_handlers,
        my_file,
        pdf_filename,
        time_start,
        time_end,
        label_yaxis,
        plot_type,
        width_inch,
        height_inch,
        keep_holding_figure=keep_holding_figure,
        continue_figure=continue_figure,
        custom_labels=custom_labels,
        plot_columns=plot_columns,
        plot_all=True)

    ##########################################################
    # Page 6
    ############################
    my_file = folder+'/'+get_file(folder, 'basalDenCa.dat')
    keep_holding_figure = False
    continue_figure = False
    label_yaxis = "basal"
    plot_type = "concentrationCalcium"
    width_inch = 10
    height_inch = 3
    custom_labels = [r'$\Delta$F/F0']
    plot_columns = [index_data_dendrite]
    fluorescence_total = 100.0  # [uM]
    # Kd  = 1.4  # [uM] - Fluo-4 in vivo
    Kd = 0.35  # [uM] - Fluo-4 in vitro [Hagen et al., 2012]
    basal_calcium = 0.1  # [uM]
    write_pdf_new_page_calcium_dye(
        pdf_file_handlers,
        my_file,
        pdf_filename,
        fluorescence_total,
        Kd,
        basal_calcium,
        time_start,
        time_end,
        label_yaxis,
        plot_type,
        width_inch,
        height_inch,
        keep_holding_figure=keep_holding_figure,
        continue_figure=continue_figure,
        custom_labels=custom_labels,
        plot_columns=plot_columns,
        plot_all=True)

    ##########################################################
    #  SPINE HEAD + NECK
    ############################
    ##########################################################
    # Page ???
    ############################
    plot_type = "voltagecalcium"
    my_file = folder+'/'+get_file(folder, 'spineheadV.dat')
    my_file2 = folder+'/'+get_file(folder, 'spineheadCa.dat')
    label_yaxis = r'spinehead'
    list_indices = get_list_indices(my_file, pattern="[1:10]")
    write_pdf_new_page_overlap_spines(
        pdf_file_handlers,
        my_file,
        my_file2,
        pdf_filename,
        time_start,
        time_end,
        label_yaxis,
        plot_type,
        list_indices,
        width_inch,
        height_inch,
    )
    ##########################################################
    # Page 1+ - plot all spinehead currents
    ############################
    # label_yaxis = "spinehead"
    # plotcurrent_in_group_spines(pdf_file_handlers, folder, 'spineheadCurrents.dat', label_yaxis,
    #                             pdf_filename,
    #                             time_start,
    #                             time_end,
    #                             list_indices
    #                             )
    ##########################################################
    # Page ???
    ############################
    plot_type = "voltagecalcium"
    my_file = folder+'/'+get_file(folder, 'spineneckV.dat')
    my_file2 = folder+'/'+get_file(folder, 'spineneckCa.dat')
    label_yaxis = r'spineneck'
    write_pdf_new_page_overlap_spines(
        pdf_file_handlers,
        my_file,
        my_file2,
        pdf_filename,
        time_start,
        time_end,
        label_yaxis,
        plot_type,
        list_indices,
        width_inch,
        height_inch)
    ##########################################################
    # Page 1+ - plot all spineneck currents
    ############################
    # label_yaxis = "spineneck"
    # plotcurrent_in_group_spines(pdf_file_handlers, folder, 'spineneckCurrents.dat', label_yaxis,
    #                             pdf_filename,
    #                             time_start,
    #                             time_end,
    #                             list_indices
    #                             )
    # FINAL
    close_pdf_file(pdf_file_handlers, pdf_filename)
    import shutil
    shutil.copy(pdf_filename, folder)
    return
    # END


def plotcurrent_in_group_Vclamp(pdf_file_handlers, folder, filename,
                                label_yaxis,
                                pdf_filename,
                                time_start, time_end):
    """
    currents are plotted using the same set of plots, so we group them here
    current are plot at whole-cell level, i.e. using soma's surface area detected from neuron.swc file
    """
    if 'Vclamp' not in folder:
        return
    else:
        swc_file = "neuron.swc"
        my_file = folder+'/'+get_file(folder, swc_file)
        # swc_tree = btmorph.STree2()
        # swc_tree.read_SWC_tree_from_file(my_file)
        # stats = btmorh.BTStats(swc_tree)
        # surface_area = stats.
        for line in open(my_file):
            line = line.strip()
            if not line.startswith("#"):
                soma_line = line
                break
        numbers = soma_line.split()
        surface_area = 4 * math.pi * math.pow(float(numbers[5]), 2)
        print("Soma surface area %s " % (surface_area))

    ##########################################################
    # Page 1 - plot all currents
    ############################
    my_file = folder+'/'+get_file(folder, filename)
    # label_yaxis = "soma"
    plot_type = "channelCurrent"
    width_inch = 10
    height_inch = 3
    convert_unit = ["current", "nA", surface_area]
    write_pdf_new_page(
        pdf_file_handlers,
        my_file,
        pdf_filename,
        time_start,
        time_end,
        label_yaxis,
        plot_type,
        width_inch,
        height_inch,
        convert_unit=convert_unit)
    ##########################################################
    # Page 1a - plot Na+ and KAf, KAs
    ############################
    my_file = folder+'/'+get_file(folder, filename)
    # label_yaxis = "soma"
    plot_type = "channelCurrent"
    width_inch = 10
    height_inch = 3
    show_channels = ["Nat", "KAf", "KAs"]
    convert_unit = ["current", "nA", surface_area]
    write_pdf_new_page(
        pdf_file_handlers,
        my_file,
        pdf_filename,
        time_start,
        time_end,
        label_yaxis,
        plot_type,
        width_inch=width_inch,
        height_inch=height_inch,
        show_channels=show_channels,
        convert_unit=convert_unit)
    ##########################################################
    # Page 1b - show only Ca2+ channel
    ############################
    my_file = folder+'/'+get_file(folder, filename)
    # label_yaxis = "soma"
    plot_type = "channelCurrent"
    width_inch = 10
    height_inch = 3
    hide_channels = ["Nat", "Nap", "KAf", "KAs", "KRP", "KIR", "BK", "SK", "Kv31", "IP3R", "RYR1", "NCX"]
    write_pdf_new_page(
        pdf_file_handlers,
        my_file,
        pdf_filename,
        time_start,
        time_end,
        label_yaxis,
        plot_type,
        width_inch=width_inch,
        height_inch=height_inch,
        hide_channels=hide_channels,
        convert_unit=convert_unit)
    ##########################################################
    # Page 1b - show only K+ channels
    ############################
    my_file = folder+'/'+get_file(folder, filename)
    # label_yaxis = "soma"
    plot_type = "channelCurrent"
    width_inch = 10
    height_inch = 3
    show_channels = ["KAf", "KAs", "KRP", "KIR", "BK", "SK", "Kv31"]
    write_pdf_new_page(
        pdf_file_handlers,
        my_file,
        pdf_filename,
        time_start,
        time_end,
        label_yaxis,
        plot_type,
        width_inch=width_inch,
        height_inch=height_inch,
        show_channels=show_channels,
        convert_unit=convert_unit)
    ##########################################################
    # Page 1c - shows PUMP, Exchangers
    ############################
    my_file = folder+'/'+get_file(folder, filename)
    # label_yaxis = "soma"
    plot_type = "channelCurrent"
    width_inch = 10
    height_inch = 3
    show_channels = ["NCX", "PMCA"]
    write_pdf_new_page(
        pdf_file_handlers,
        my_file,
        pdf_filename,
        time_start,
        time_end,
        label_yaxis,
        plot_type,
        width_inch=width_inch,
        height_inch=height_inch,
        show_channels=show_channels,
        convert_unit=convert_unit)


if __name__ == "__main__":
    # OPTIONS = {0: 'plot_soma', 100: 'plot_case0', 2: 'plot_case2', 3: 'plot_case3',
    #            4: 'plot_case4', 5: 'plot_case5', 6: 'plot_case6',
    #            7: 'plot_case7', 8: 'plot_case8', 9: 'plot_case9_adv_PDF',
    #            10: 'simpleSpine'}
    # OPTIONS = {0: plot_soma, 100: plot_case0, 2: plot_case2, 3: plot_case3,
    #            4: plot_case4, 5: plot_case5, 6: plot_case6,
    #            7: plot_case7, 8: plot_case8, 9: plot_case9_adv_PDF,
    #            10: simpleSpine}
    OPTIONS = {9: plot_case9_adv_pdf, 11: plot_msn_microdomain_pdf,
               12: plot_msn_microdomain_pdf_repeats,
               13: plot_test}
    OPTIONS.get(12, plot_case9_adv_pdf)()
    # NOTE:
    # case 3: no neuron, only bouton + spinehead : use =4 is better
    # case 4: no neuron, only bouton + spinehead, with NMDAR current recording
    # case 5: 1 neuron, only bouton + spinehead,  with NMDAR current recording
    # case 6: automatic detect filename in MPI-scenarios
    # case 7: automatic detect filename in MPI-scenarios (neuron + spines)
    # case 8: automatic detect filename in MPI-scenarios (neuron + spines) - fix
    #    to 24 MPI processes
    # case 9:
    #    plot_case9()
    #    plot_case9_adv()
    #    plot_case9_adv_currentOnly()
    #    plot_case9_adv_PDF()
    #    plot_case9_adv_original()
