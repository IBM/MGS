import logging
# import re
# import btmorph

# import numpy as np
import os.path
# import csv
# import fnmatch
from file_utils import write_pdf_new_page, write_pdf_new_page_overlap # NOQA
from file_utils import write_pdf_new_page_currents  # NOQA
from file_utils import write_pdf_new_page_calcium_dye  # NOQA
from file_utils import create_new_pdf_file, get_file, close_pdf_file  # NOQA
from file_utils import get_working_folder, get_pdf_folder  # NOQA


def plot_case9_adv_pdf():
    """
    Plot figures to PDF file
    """
    # more comprehensive deal with multiple processes I/O
    # with spines
    # output directly to PDF file

    ##########
    # USER-DEFINED SECTION
    time_start = 0.0
    # time_start = 390.0
    time_end = -1.0
    # time_end = 470.0
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
    # Page 3
    ############################
    my_file = folder+'/'+get_file(folder, 'axonAISCurrents.dat')
    label_yaxis = "axon-AIS"
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
    # Page 4
    ############################
    my_file = folder+'/'+get_file(folder, 'axonAISV.dat')
    my_file2 = folder+'/'+get_file(folder, 'axonAISCa.dat')
    label_yaxis = "axon-AIS"
    plot_type = "voltagecalcium"
    width_inch = 10
    height_inch = 3
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
    # Page 5
    ############################
    my_file = folder+'/'+get_file(folder, 'basalDen11xbrOrder2Currents.dat')
    label_yaxis = "basalDen11xbrOrder2"
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
    # Page 6
    ############################
    my_file = folder+'/'+get_file(folder, 'basalDen11xbrOrder2V.dat')
    my_file2 = folder+'/'+get_file(folder, 'basalDen11xbrOrder2Ca.dat')
    label_yaxis = "basalDen11xbrOrder2"
    plot_type = "voltagecalcium"
    width_inch = 10
    height_inch = 3
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
    # Page 6
    ############################
    my_file = folder+'/'+get_file(folder, 'basalDen11xbrOrder3Currents.dat')
    label_yaxis = "basalDen11xbrOrder3"
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
    # Page 7
    ############################
    my_file = folder+'/'+get_file(folder, 'basalDen11xbrOrder3V.dat')
    my_file2 = folder+'/'+get_file(folder, 'basalDen11xbrOrder3Ca.dat')
    label_yaxis = "basalDen11xbrOrder3"
    plot_type = "voltagecalcium"
    width_inch = 10
    height_inch = 3
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
    # Page 8
    ############################
    my_file = folder+'/'+get_file(folder, 'basalDen11xbrOrder4Currents.dat')
    label_yaxis = "basalDen11xbrOrder4"
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
    # Page 9
    ############################
    my_file = folder+'/'+get_file(folder, 'basalDen11xbrOrder4V.dat')
    my_file2 = folder+'/'+get_file(folder, 'basalDen11xbrOrder4Ca.dat')
    label_yaxis = "basalDen11xbrOrder4"
    plot_type = "voltagecalcium"
    width_inch = 10
    height_inch = 3
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
    # Page 10
    ############################
    my_file = folder+'/'+get_file(folder, 'basalDen11xbrOrder5Currents.dat')
    label_yaxis = "basalDen11xbrOrder5"
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
    # Page 11
    ############################
    my_file = folder+'/'+get_file(folder, 'basalDen11xbrOrder5V.dat')
    my_file2 = folder+'/'+get_file(folder, 'basalDen11xbrOrder5Ca.dat')
    label_yaxis = "basalDen11xbrOrder5"
    plot_type = "voltagecalcium"
    width_inch = 10
    height_inch = 3
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
    # FINAL
    close_pdf_file(pdf_file_handlers, pdf_filename)
    import shutil
    shutil.copy(pdf_filename, folder)
    return
    # END
