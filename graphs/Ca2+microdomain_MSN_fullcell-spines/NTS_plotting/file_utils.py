"""Example Google style docstrings.

This module demonstrates documentation as specified by the `Google Python
Style Guide`_. Docstrings may extend over multiple lines. Sections are created
with a section header and a colon followed by a block of indented text.

Example:
    Examples can be given using either the ``Example`` or ``Examples``
    sections. Sections support any reStructuredText formatting, including
    literal blocks::

        $ python example_google.py

Section breaks are created by resuming unindented text. Section breaks
are also implicitly created anytime a new section starts.

Attributes:
    module_level_variable1 (int): Module level variables may be documented in
        either the ``Attributes`` section of the module docstring, or in an
        inline docstring immediately following the variable.

        Either form is acceptable, but the two should not be mixed. Choose
        one convention to document module level variables and be consistent
        with it.

Todo:
    * For module TODOs
    * You have to also use ``sphinx.ext.todo`` extension

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""
from __future__ import print_function
import logging
import os.path
import os
import sys
import datetime
import fnmatch
import argparse
import re
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt  # NOQA
from matplotlib.backends.backend_pdf import PdfPages  # NOQA
from globals import COLORS, LINE_STYLES
import globals
import pandas as pd
# , _storedFig, _storedAxes, _offset  # NOQA
__author__ = "Hoang Trong Minh Tuan"
__copyright__ = "Copyright 2016, IBM"
__credits__ = []  # list of people who file pugs
__version__ = "0.1"
__maintainer__ = "Hoang Trong Minh Tuan"
__email__ = "tmhoangt@us.ibm.com"
__status__ = "Prototype"  # (Prototype, Development, Production)
logger = logging.getLogger(__name__)  # pylint: disable=locally-disabled, invalid-name


def parse():
    """
    parse the command-line arguments
    """
    # global parser
    from argparse import RawTextHelpFormatter
    _parser_description = """
    plot[Vm, Ca, Currents] in multiple windows.
    """
    _parser_epilog = """
    Examples: \n
        1. plot data in <default>/Myfolder
    python plot_currents.py folder MyFolder \n
        2. plot data in /home/yourname/Myfolder
    python plot_currents.py folder MyFolder  -loc /home/yourname \n
        3. plot data in /home/yourname/msn0_Wolf_triggersoma_2016-10-18-14788282
    python plot_currents.py protocol 2016-10-18-14788282 Wolf 1
        """
    _subparser_protocol_description = """
        Pass in the components that help to build the data folder
        NOTE: The output folder is expected to be in this format
        <data_location>/<morphology>_<AuthorName>_<protocol>_<date>
        \n
        """
    _subparser_folder_description = """
        Pass in the complete name of the data folder
        NOTE: The output folder is expected to be
        <data_location>/<complete_name>
        """
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("-locData", dest='location',
                               default="./data", help="the location to data-folder")
    # parent_parser.add_argument("-locPDF", dest='locationPDF',
    #             default="/data/tmhoangt/PDF_NTS_OUTPUT/",
    #             help="the location where PDF plotting-file is saved")
    parent_parser.add_argument("-locPDF", dest='locationPDF',
                               default="/home/tmhoangt/chdi_common/tuan/PDF_NTS_OUTPUT/",
                               help="the location where PDF plotting-file is saved")

    parser = argparse.ArgumentParser(description=_parser_description,
                                     epilog=_parser_epilog,
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('--version', action='version', version='1.0.1')
    subparsers = parser.add_subparsers(dest="subparsers_name")

    parser_folder = subparsers.add_parser('folder',
                                          help=_subparser_folder_description,
                                          parents=[parent_parser])
    parser_folder.add_argument("folderName",
                               help="full name of the folder with data to be plotted")
    parser_folder.set_defaults(which='folder')

    parser_protocol = subparsers.add_parser('protocol',
                                            help=_subparser_protocol_description,
                                            description=_subparser_protocol_description,
                                            parents=[parent_parser])
    parser_protocol.add_argument("morphology", default="",
                                 help="(optional) name of morph, e.g. hay1")
    parser_protocol.add_argument("author", default="",
                                 help="(optional) name of model's author, e.g. Wolf")
    parser_protocol.add_argument("number", default=-1,
                                 help="index of simulation protocol (-1: create folder only; 0 = rest; 1 = inject soma; 2= inject shaft with dual-exp EPSP-like current; 3 = inject a particular presynaptic neuron; 4 = like 3, but at a distal region; 5 = trigger soma then at distal end (within a small window of time); 6 = trigger soma, then another spine (within a window of time))")  # NOQA
    # parser_protocol.add_argument("date", default="", help="date of data, e.g. 2016-08-30-1472587619")
    # parser_protocol.set_defaults(func=plot_case9a)
    parser_protocol.add_argument("-extension", default="",
                                 help="an extension to uniquely identify the folder, e.g. 2016-08-30-1472587619")
    parser_protocol.set_defaults(which='protocol')

    # global args
    args = parser.parse_args()
    return args


def get_file(folder, filename_prefix):
    """
    as the file is writen with MPI-rank information which can be different
    depending upon the mpiexec configuration
    Input is the location and filename without MPI rank
    RETURN:
        return the exact file name
    """
    for file in os.listdir(folder):
        if fnmatch.fnmatch(file, filename_prefix+'*'):
            return file
    logging.info("File %s not present", os.path.join(folder, filename_prefix))
    return ""


def get_working_folder():
    """
    Return the folder with data to be plot
    """
    args = parse()
    logging.debug(args.subparsers_name)
    logging.debug(args)

    main_folder = args.location

    if args.subparsers_name == 'folder':
        folder_fullpath = os.path.join(main_folder, args.folderName)
        foldername_only = args.folderName
    elif args.subparsers_name == 'protocol':
        morph = args.morphology.strip('" ')
        extension = args.extension
        author = args.author
        if extension != "":
            extension = "-" + extension
        if morph != "" and not morph.endswith("_"):
            morph += "_"
        if author != "":
            author += "_"
        map_folders = {}
        map_folders[0] = morph + author + 'rest' + extension  # NOQA
        map_folders[1] = morph + author + 'triggersoma' + extension  # NOQA
        map_folders[2] = morph + author + 'triggershaft' + extension  # NOQA
        map_folders[3] = morph + author + 'triggerspine' + extension  # NOQA
        map_folders[4] = morph + author + 'triggerdistalspines' + extension  # NOQA
        map_folders[5] = morph + author + 'triggersoma_then_distalspines' + extension  # NOQA
        map_folders[6] = morph + author + 'case06' + extension  # NOQA
        map_folders[7] = morph + author + 'triggeraxon' + extension  # NOQA
        map_folders[8] = morph + author + 'Vclamp' + extension  # NOQA
        map_folders[9] = morph + author + 'triggersoma' + extension  # NOQA
        # tmpMap = {}
        # date = args.date
        protocol = args.number
        # for key, val in map_folders.iteritems():
        #   # print(val.replace('May29', date))
        #   tmpMap[key] = val.replace('May29', date)
        # map_folders = tmpMap
        if int(protocol) == -1:  # first arg should be the protocol
            # create folder purpose (if not exist)
            for key, value in map_folders.iteritems():
                folder_fullpath = os.path.join(main_folder, value)
                if not os.path.exists(folder_fullpath):
                    os.makedirs(folder_fullpath)
            sys.exit("Folders created")
        foldername_only = map_folders[int(protocol)]
        folder_fullpath = os.path.join(main_folder, foldername_only)
        logging.info("Plot folder " + folder_fullpath)
    else:
        logging.info("Unknown method")
        sys.exit("")
    # print(args)
    return main_folder, foldername_only


def get_pdf_folder():
    """
    Return the folder where PDF plot file is saved
    """
    args = parse()
    logging.debug(args.subparsers_name)
    logging.debug(args)

    pdf_folder = args.locationPDF
    if not os.path.exists(pdf_folder):
        os.makedirs(pdf_folder)
    return pdf_folder


def create_new_pdf_file(pdf_file_handlers, pdf_filename):
    """
    create a new PDF file
    """
    pdf = PdfPages(pdf_filename)
    pdf_file_handlers[pdf_filename] = pdf


def close_pdf_file(pdf_file_handlers, pdf_filename):
    """
    close the currently opening pdf files via the handlers
    """
    _pdf_subject = 'How to create a multipage pdf file and set its metadata'
    _pdf_title = 'Multipage PDF Example'
    _pdf_author = u'Tuan M. Hoang Trong'
    _pdf_keywords = 'PdfPages multipage keywords author title subject'
    pdf = pdf_file_handlers[pdf_filename]
    # We can also set the file's metadata via the PdfPages object:
    document = pdf.infodict()
    document['Title'] = _pdf_title
    document['Author'] = _pdf_author
    document['Subject'] = _pdf_subject
    document['Keywords'] = _pdf_keywords
    document['CreationDate'] = datetime.datetime(2009, 11, 13)
    document['ModDate'] = datetime.datetime.today()
    pdf.close()


def write_to_pdf_simple(pdf_file_handlers, pdf_filename, newFile=True, completeWriting=True):
    """
    write data to a pdf file - simple way
    """
    _pdf_subject = 'How to create a multipage pdf file and set its metadata'
    _pdf_title = 'Multipage PDF Example'
    _pdf_author = u'Tuan M. Hoang Trong'
    _pdf_keywords = 'PdfPages multipage keywords author title subject'
    if newFile is True:
        pdf = PdfPages(pdf_filename)
        pdf_file_handlers[pdf_filename] = pdf
    else:
        pdf = pdf_file_handlers[pdf_filename]
    plt.figure(figsize=(3, 3))  # size of page 1
    plt.plot(range(7), [3, 1, 4, 1, 5, 9, 2], 'r-o')
    plt.title('Page One')
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()

    plt.rc('text', usetex=True)
    plt.figure(figsize=(8, 6))  # size of pageg 2
    data_x = np.arange(0, 5, 0.1)
    plt.plot(data_x, np.sin(data_x), 'b-')
    plt.title('Page Two')
    # pdf.attach_note("plot of sin(data_x)")  # you can add a pdf note to
    #                                # attach metadata to a page
    pdf.savefig()
    plt.close()

    plt.rc('text', usetex=False)
    fig = plt.figure(figsize=(4, 5))
    plt.plot(data_x, data_x*data_x, 'ko')
    plt.title('Page Three')
    pdf.savefig(fig)  # or you can pass a Figure object to pdf.savefig
    plt.close()

    # We can also set the file's metadata via the PdfPages object:
    document = pdf.infodict()
    document['Title'] = _pdf_title
    document['Author'] = _pdf_author
    document['Subject'] = _pdf_subject
    document['Keywords'] = _pdf_keywords
    document['CreationDate'] = datetime.datetime(2009, 11, 13)
    document['ModDate'] = datetime.datetime.today()
    if completeWriting is True:
        pdf.close()


def write_pdf_new_page(pdf_file_handlers,
                       my_file, pdf_filename,
                       time_start, time_end, label_yaxis,
                       plot_type, width_inch=3, height_inch=3,
                       keep_holding_figure=False, continue_figure=False,
                       custom_labels=None, offset_xaxis=None,
                       hide_channels=[],
                       show_channels=[],
                       plot_columns=[],
                       plot_all=False,
                       convert_unit=["current", "pA/um^2"]):
    """
    my_file = data file (column-based data; with 2 or many columns)
              the first column is typically time-information
              (but can also something else)
    time_start
    time_end    = range of data to plot
                (can be time or something else)
                [assume the first column in data-file is this information]
    label_yaxis     = label o Y-axis
    plot_type   = the type of data [help to customize the data-read]
    width_inch
    height_inch      = height in inches
    keep_holding_figure = if True, then wait for another data to be plotted
        it will use: _storedFig, _storedAxes, _offset
        to draw new lines on the existing plot
    continue_figure   = if True, then use the Figure stored in _storedFig, _storedAxes, _offset
    custom_labels      = user-defined label, i.e. not reading from first-line header
                    (use for data from the second-column beyond)
    hide_channels = list of channel names that are skipped (only for plot_type=="channelCurrent")
    show_channels = list of channel names that are plotted (only for plot_type=="channelCurrent")
                NOTE: hide_channels and show_channels are EXCLUSIVE
    plot_columns = if [], then this parameter has no role
                   if not empty, then this parameter indicate the column-indices of data to be plotted
                   (when plot_all==True)
    plot_all     = if False, then this parameter has no role
                   if True, then it performs plotting all columns (with an option to limit
                   what column indices using plot_columns option)
    convert_unit  = ["current", "pA/um^2"]  -- default
                    ["current", "uA/cm^2"]
                    ["current", "mA/cm^2"]
                    ["current", "nA", surface_area]  - whole-cell
                    [default is pA/um^2 for current-density]
                    NOTE: surface_area = [um^2]
    NOTE: The axes range are based on the data on first file if
              multiple files are used
    """
    _acceptedPlotType = ["channelCurrent", "voltage", "concentrationCalcium", "currentStim"]
    if plot_type not in _acceptedPlotType:
        logging.error("Not valid plot_type")
        logging.error("... using " + plot_type)
        logging.error("...accepted: " + ', '.join(_acceptedPlotType))
        return

    if pdf_filename not in pdf_file_handlers:
        pdf = PdfPages(pdf_filename)
        pdf_file_handlers[pdf_filename] = pdf
    else:
        pdf = pdf_file_handlers[pdf_filename]

    fig = None
    axes = None
    if continue_figure is False:
        if os.path.isfile(my_file):
            fig = plt.figure(figsize=(width_inch, height_inch))  # size of page 1 in inches
            axes = fig.add_subplot(111)
            globals.storedFig = fig
            globals.storedAxes = axes
            globals.offset = 0
    else:
        fig = globals.storedFig
        axes = globals.storedAxes
        globals.offset += 1

    if not os.path.isfile(my_file):
        logging.info("File %s not present", my_file)
        if keep_holding_figure is False and continue_figure is True:
            axes.set_ylabel(label_yaxis)
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()
        return

    num_columns = 1
    num_skipped_row = 1
    if plot_type == "channelCurrent":
        file = open(my_file)
        lines = file.readlines()
        # channel_names = re.split(",", lines[1])
        channel_names = re.split(",", re.sub(r'[\s+]', '', lines[1]))
        del channel_names[-1], channel_names[0]
        # print lines[1]
        num_skipped_row = 2
    elif plot_type == "voltage":
        channel_names = ["Vm"]
        if custom_labels is not None:
            if isinstance(custom_labels, list):
                channel_names = custom_labels
            else:
                logging.error("expect a list for 'custom_labels' argument")
                return
    elif plot_type == "concentrationCalcium":
        channel_names = ["Calcium"]
        if custom_labels is not None:
            if isinstance(custom_labels, list):
                channel_names = custom_labels
            else:
                logging.error("expect a list for 'custom_labels' argument")
                return
    elif plot_type == "currentStim":
        channel_names = ["Istim"]
        if custom_labels is not None:
            if isinstance(custom_labels, list):
                channel_names = custom_labels
            else:
                logging.error("expect a list for 'custom_labels' argument")
                return

    # print(channel_names)  # NOQA
    num_columns = len(channel_names)  # NOQA
    try:
        arrays = np.loadtxt(my_file,
                            skiprows=num_skipped_row,
                            # usecols=(0, 1)
                            )
    except ValueError:
        print("Error reading file %s with skip = %d rows" % (my_file, num_skipped_row))
        return
    arraysT = np.transpose(arrays)

    # CONDITION 1: first column is the X-axis (e.g. time)
    t = arraysT[0]
    if offset_xaxis is not None:
        t = t + offset_xaxis

    hide = []   # decide what channel to hide
    if plot_type == "channelCurrent":
        hide = hide_channels
        # hide = ['Nat', 'KAf', 'KDR', 'Cah', 'BK', 'SK']
        # hide = ['Nat', 'KAf', 'KDR', 'BK', 'SK']
        # pass
    if hide_channels and show_channels:
        logging.error("Either hide_channels xor show_channels but not both")
        return
    #############
    # Identify the array-index based on the defined time range
    idx_start = next(x[0] for x in enumerate(t) if x[1] >= time_start)
    if time_end == -1.0:
        idx_end = len(t)-1
        time_end = t[idx_end]
    else:
        idx_end = next(x[0] for x in enumerate(t) if x[1] >= time_end)
    # END
    #################
    arraysT = np.delete(arraysT, 0, 0)
    # arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
    # print arr
    # arr = np.delete(arr,0,1)
    # print arr
    # return
    # print t
    # print num_columns
    value_min = 200000000.0
    value_max = -value_min
    index_ii = 0
    if (plot_type in ["channelCurrent", "voltage", "concentrationCalcium"]):
        for ydata in arraysT:
            if plot_type in ["channelCurrent"] and convert_unit[0:2] == ["current", "uA/cm^2"]:
                ydata = 100 * ydata  # [convert from pA/um^2 to uA/cm^2]
            elif plot_type in ["channelCurrent"] and convert_unit[0:2] == ["current", "mA/cm^2"]:
                ydata = 0.1 * ydata  # [convert from pA/um^2 to mA/cm^2]
            elif plot_type in ["channelCurrent"] and convert_unit[0:2] == ["current", "nA"]:
                ydata = ydata * convert_unit[2] * 1e-3  # [convert from pA/um^2 to nA]
            # plot (t, col)
            # legend(time, channel_names[i])
            color = COLORS[(index_ii+globals.offset) % len(COLORS)]
            style = LINE_STYLES[((index_ii / len(COLORS)) % len(LINE_STYLES))]
            # axarr[gr, gc].plot(t, ydata, color, label=channel_names[index_ii])
            if plot_all is True:
                if plot_columns:
                    if index_ii + 2 in plot_columns:
                        axes.plot(t, ydata, linestyle=style,
                                  color=color, label=channel_names[index_ii % num_columns])
                        axes.legend(ncol=2, prop={'size': 8},
                                    loc=4, handlelength=3)  # shadow=True, fancybox=True, loc="upper left", bbox_to_anchor=[0,1]
                        value_min = min(np.amin(ydata[idx_start:idx_end]), value_min)
                        value_max = max(np.amax(ydata[idx_start:idx_end]), value_max)
                else:
                    axes.plot(t, ydata, linestyle=style,
                              color=color, label=channel_names[index_ii % num_columns])
                    axes.legend(ncol=2, prop={'size': 8},
                                loc=4, handlelength=3)  # shadow=True, fancybox=True, loc="upper left", bbox_to_anchor=[0,1]
                    value_min = min(np.amin(ydata[idx_start:idx_end]), value_min)
                    value_max = max(np.amax(ydata[idx_start:idx_end]), value_max)
                # NOTE: handlelength ensure proper dotted-line show in legend
                # x = np.arange(0, 5, 0.1)
                # axes.plot(t, ydata, 'b-')
                # value_min = min(np.amin(ydata[idx_start:idx_end]), value_min)
                # value_max = max(np.amax(ydata[idx_start:idx_end]), value_max)
            else:
                if (show_channels and channel_names[index_ii] in show_channels) or \
                        (hide and channel_names[index_ii] not in hide) or \
                        (not hide and not show_channels):
                    axes.plot(t, ydata, linestyle=style, color=color, label=channel_names[index_ii])
                    axes.legend(ncol=2, prop={'size': 8}, loc=4, handlelength=3)  # shadow=True, fancybox=True, loc="upper left", bbox_to_anchor=[0,1]
                    # NOTE: handlelength ensure proper dotted-line show in legend
                    # x = np.arange(0, 5, 0.1)
                    # axes.plot(t, ydata, 'b-')
                    value_min = min(np.amin(ydata[idx_start:idx_end]), value_min)
                    value_max = max(np.amax(ydata[idx_start:idx_end]), value_max)
            # print len(t), len(ydata)
            index_ii += 1
        # print i
        # print value_min, value_max
        # axes.set_ylim(bottom=value_min - 0.05 * max(0.1, abs(value_min)))
        # axes.set_ylim(top=value_max + 0.05 * max(0.1, abs(value_max)))
        axes.set_ylim(bottom=value_min - 0.05 * abs(value_min))
        axes.set_ylim(top=value_max + 0.05 * abs(value_max))
        axes.set_xlim(left=time_start, right=time_end)
        if plot_type == "channelCurrent":
            if convert_unit[0:2] == ["current", "uA/cm^2"]:
                axes.set_title(r"time (ms) vs. currents ($\mu$A/cm$^2$)")
            elif convert_unit[0:2] == ["current", "mA/cm^2"]:
                axes.set_title(r"time (ms) vs. currents (pA/cm$^2$)")
            elif convert_unit[0:2] == ["current", "pA/um^2"]:
                axes.set_title(r"time (ms) vs. currents (pA/$\mu$m$^2$)")
            elif convert_unit[0:2] == ["current", "nA"]:
                axes.set_title(r"time (ms) vs. currents (nA)")
        if plot_type == "voltage":
            axes.set_title("time (ms) vs. voltage (mV)")

    # other type of data
    # ...
    if (plot_type in ["currentStim"]):
        for ydata in arraysT:
            color = COLORS[(index_ii+globals.offset) % len(COLORS)]
            style = LINE_STYLES[((index_ii / len(COLORS)) % len(LINE_STYLES))]
            axes.plot(t, ydata, linestyle=style, color=color, label=channel_names[index_ii])
            axes.legend(ncol=2, prop={'size': 8}, loc=4, handlelength=3)  # shadow=True, fancybox=True, loc="upper left", bbox_to_anchor=[0,1]
            # NOTE: handlelength ensure proper dotted-line show in legend
            # x = np.arange(0, 5, 0.1)
            # axes.plot(t, ydata, 'b-')
            value_min = min(np.amin(ydata[idx_start:idx_end]), value_min)
            value_max = max(np.amax(ydata[idx_start:idx_end]), value_max)
        axes.set_ylim(bottom=value_min - 0.05 * max(0.1, abs(value_min)))
        axes.set_ylim(top=value_max + 0.05 * max(0.1, abs(value_max)))
        axes.set_xlim(left=time_start, right=time_end)
        if plot_type == "currentStim":
            if convert_unit[0:2] == ["current", "pA"]:
                axes.set_title(r"time (ms) vs. currents (pA)")

    #
    if keep_holding_figure is False:
        axes.set_ylabel(label_yaxis)
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()
    return


def write_pdf_new_page_currents(pdf_file_handlers,
                                my_file, pdf_filename,
                                time_start, time_end, label_yaxis,
                                plot_type, width_inch=3, height_inch=3,
                                keep_holding_figure=False, continue_figure=False,
                                custom_labels=None, offset_xaxis=None,
                                hide_channels=[],
                                show_channels=[],
                                plot_columns=[],
                                plot_location_index=[],
                                plot_all=False,
                                convert_unit=True):
    """
    my_file = data file (column-based data; with 2 or many columns)
              the first column is typically time-information
              (but can also something else)
    time_start
    time_end    = range of data to plot
                (can be time or something else)
                [assume the first column in data-file is this information]
    label_yaxis     = label o Y-axis
    plot_type   = the type of data [help to customize the data-read]
    width_inch
    height_inch      = height in inches
    keep_holding_figure = if True, then wait for another data to be plotted
        it will use: _storedFig, _storedAxes, _offset
        to draw new lines on the existing plot
    continue_figure   = if True, then use the Figure stored in _storedFig, _storedAxes, _offset
    custom_labels      = user-defined label, i.e. not reading from first-line header
                    (use for data from the second-column beyond)
    hide_channels = list of channel names that are skipped (only for plot_type=="channelCurrent")
    show_channels = list of channel names that are plotted (only for plot_type=="channelCurrent")
                NOTE: hide_channels and show_channels are EXCLUSIVE
    plot_columns = if [], then this parameter has no role
                   if not empty, then this parameter indicate the column-indices of data to be plotted
                   (when plot_all==True)
    plot_all     = if False, then this parameter has no role
                   if True, then it performs plotting all columns (with an option to limit
                   what column indices using plot_columns option)
    plot_location_index = when recording, there is a chance we record the whole branch or multiple branches
                   [uggg, here to tell as different branches has different #compartments]
    NOTE: The axes range are based on the data on first file if
              multiple files are used
    """
    _acceptedPlotType = ["channelCurrent", "voltage", "concentrationCalcium"]
    if plot_type not in _acceptedPlotType:
        logging.error("Not valid plot_type")
        logging.error("... using " + plot_type)
        logging.error("...accepted: " + ', '.join(_acceptedPlotType))
        return

    if pdf_filename not in pdf_file_handlers:
        pdf = PdfPages(pdf_filename)
        pdf_file_handlers[pdf_filename] = pdf
    else:
        pdf = pdf_file_handlers[pdf_filename]

    fig = None
    axes = None
    if continue_figure is False:
        fig = plt.figure(figsize=(width_inch, height_inch))  # size of page 1 in inches
        axes = fig.add_subplot(111)
        globals.storedFig = fig
        globals.storedAxes = axes
        globals.offset = 0
    else:
        fig = globals.storedFig
        axes = globals.storedAxes
        globals.offset += 1

    if not os.path.isfile(my_file):
        logging.info("File %s not present", my_file)
        if keep_holding_figure is False:
            axes.set_ylabel(label_yaxis)
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()
        return

    num_columns = 1
    num_skipped_row = 1
    stride = 1
    if plot_type == "channelCurrent":
        file = open(my_file)
        lines = file.readlines()
        # channel_names = re.split(",", lines[1])
        channel_names = re.split(",", re.sub(r'[\s+]', '', lines[1]))
        del channel_names[-1], channel_names[0]
        # print lines[1]
        num_skipped_row = 2
        # unique_channel_names = set()
        # for item in iterable:
        #         unique_channel_names.add(item)
        number_compartments = channel_names.count(channel_names[0])
        stride = number_compartments
        print("stride=", stride)

    elif plot_type == "voltage":
        channel_names = ["Vm"]
        if custom_labels is not None:
            if isinstance(custom_labels, list):
                channel_names = custom_labels
            else:
                logging.error("expect a list for 'custom_labels' argument")
                return
    elif plot_type == "concentrationCalcium":
        channel_names = ["Calcium"]
        if custom_labels is not None:
            if isinstance(custom_labels, list):
                channel_names = custom_labels
            else:
                logging.error("expect a list for 'custom_labels' argument")
                return

    print(channel_names)  # NOQA
    num_columns = len(channel_names)  # NOQA
    try:
        arrays = np.loadtxt(my_file,
                            skiprows=num_skipped_row,
                            # usecols=(0, 1)
                            )
    except ValueError:
        print("Error reading file %s with skip = %d rows" % (my_file, num_skipped_row))
        return
    arraysT = np.transpose(arrays)

    # CONDITION 1: first column is the X-axis (e.g. time)
    t = arraysT[0]
    if offset_xaxis is not None:
        t = t + offset_xaxis

    hide = []   # decide what channel to hide
    if plot_type == "channelCurrent":
        hide = hide_channels
        # hide = ['Nat', 'KAf', 'KDR', 'Cah', 'BK', 'SK']
        # hide = ['Nat', 'KAf', 'KDR', 'BK', 'SK']
        # pass
    if hide_channels and show_channels:
        logging.error("Either hide_channels xor show_channels but not both")
        return
    #############
    # Identify the array-index based on the defined time range
    idx_start = next(x[0] for x in enumerate(t) if x[1] >= time_start)
    if time_end == -1.0:
        idx_end = len(t)-1
        time_end = t[idx_end]
    else:
        idx_end = next(x[0] for x in enumerate(t) if x[1] >= time_end)
    # END
    #################
    arraysT = np.delete(arraysT, 0, 0)
    # arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
    # print arr
    # arr = np.delete(arr,0,1)
    # print arr
    # return
    # print t
    # print num_columns
    value_min = 200000000.0
    value_max = -value_min
    index_ii = 0
    data_column = -1
    if (plot_type in ["channelCurrent", "voltage", "concentrationCalcium"]):
        counter = stride
        for ydata in arraysT:
            if plot_type in ["channelCurrent"] and convert_unit:
                ydata = 100 * ydata  # [convert from pA/um^2 to uA/cm^2]
            data_column += 1
            if counter < stride:
                counter += 1
                continue
            else:
                counter = 1

            # plot (t, col)
            # legend(time, channel_names[i])
            color = COLORS[(index_ii+globals.offset) % len(COLORS)]
            style = LINE_STYLES[((index_ii / len(COLORS)) % len(LINE_STYLES))]
            # axarr[gr, gc].plot(t, ydata, color, label=channel_names[index_ii])
            if plot_all is True:
                if plot_columns:
                    if (data_column + 2 in plot_columns):
                        axes.plot(t, ydata, linestyle=style,
                                  color=color, label=channel_names[data_column % num_columns])
                        axes.legend(ncol=2, prop={'size': 8},
                                    loc=4, handlelength=3)  # shadow=True, fancybox=True, loc="upper left", bbox_to_anchor=[0,1]
                        value_min = min(np.amin(ydata[idx_start:idx_end]), value_min)
                        value_max = max(np.amax(ydata[idx_start:idx_end]), value_max)
                else:
                    axes.plot(t, ydata, linestyle=style,
                              color=color, label=channel_names[data_column % num_columns])
                    axes.legend(ncol=2, prop={'size': 8},
                                loc=4, handlelength=3)  # shadow=True, fancybox=True, loc="upper left", bbox_to_anchor=[0,1]
                    value_min = min(np.amin(ydata[idx_start:idx_end]), value_min)
                    value_max = max(np.amax(ydata[idx_start:idx_end]), value_max)
                # NOTE: handlelength ensure proper dotted-line show in legend
                # x = np.arange(0, 5, 0.1)
                # axes.plot(t, ydata, 'b-')
                # value_min = min(np.amin(ydata[idx_start:idx_end]), value_min)
                # value_max = max(np.amax(ydata[idx_start:idx_end]), value_max)
            else:
                if (show_channels and channel_names[data_column] in show_channels) or \
                        (hide and channel_names[data_column] not in hide) or \
                        (not hide and not show_channels):
                    axes.plot(t, ydata, linestyle=style, color=color, label=channel_names[data_column])
                    axes.legend(ncol=2, prop={'size': 8}, loc=4, handlelength=3)  # shadow=True, fancybox=True, loc="upper left", bbox_to_anchor=[0,1]
                    # NOTE: handlelength ensure proper dotted-line show in legend
                    # x = np.arange(0, 5, 0.1)
                    # axes.plot(t, ydata, 'b-')
                    value_min = min(np.amin(ydata[idx_start:idx_end]), value_min)
                    value_max = max(np.amax(ydata[idx_start:idx_end]), value_max)
            # print len(t), len(ydata)
            index_ii += 1
        # print i
        # print value_min, value_max
        axes.set_ylim(bottom=value_min - 0.05 * max(0.1, abs(value_min)))
        axes.set_ylim(top=value_max + 0.05 * max(0.1, abs(value_max)))
        axes.set_xlim(left=time_start, right=time_end)
        if plot_type == "channelCurrent":
            if convert_unit:
                axes.set_title(r"time (ms) vs. currents ($\mu$A/cm$^2$)")
            else:
                axes.set_title(r"time (ms) vs. currents (pA/$\mu$m$^2$)")
        if plot_type == "voltage":
            axes.set_title("time (ms) vs. voltage (mV)")

    # other type of data
    # ...

    #
    if keep_holding_figure is False:
        axes.set_ylabel(label_yaxis)
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()
    return


def write_pdf_new_page_overlap(pdf_file_handlers,
                               my_file, my_file2,
                               pdf_filename, time_start, time_end,
                               label_yaxis, plot_type, width_inch=3, height_inch=3):
    """
    my_file = data file (column-based data; with 2 or many columns)
              the first column is typically time-information
              (but can also something else)
    my_file2 = data file (column-based data; with 2 or many columns)
              the first column is typically time-information
              (but can also something else)
    time_start
    time_end    = range of data to plot
                (can be time or something else)
                [assume the first column in data-file is this information]
    label_yaxis     = label o Y-axis
    plot_type   = the type of data [help to customize the data-read]
    width_inch
    height_inch      = height in inches
    """
    if not os.path.isfile(my_file):
        logging.info("File %s not present", my_file)
        return

    _acceptedPlotType = ["channelCurrent", "voltage", "concentrationCalcium",
                         "voltagecalcium"]
    if plot_type not in _acceptedPlotType:
        logging.info("Not valid plot_type")
        logging.info("... using " + plot_type)
        logging.info("...accepted " + _acceptedPlotType)
        return

    pdf = None
    if pdf_filename not in pdf_file_handlers:
        pdf = PdfPages(pdf_filename)
        pdf_file_handlers[pdf_filename] = pdf
    else:
        pdf = pdf_file_handlers[pdf_filename]
    fig = plt.figure(figsize=(width_inch, height_inch))  # size of page 1 in inches
    # axes = fig.add_subplot(111)
    axes = fig.gca()

    num_columns = 1
    num_skipped_row = 1
    if plot_type == "channelCurrent":
        file = open(my_file)
        lines = file.readlines()
        # channel_names = re.split(",", lines[1])
        channel_names = re.split(",", re.sub(r'[\s+]', '', lines[1]))
        del channel_names[-1], channel_names[0]
        # print lines[1]
        # print channel_names
        num_skipped_row = 2
    elif plot_type == "voltagecalcium":
        channel_names = ["Vm"]

    num_columns = len(channel_names)
    arrays = np.loadtxt(my_file,
                        skiprows=num_skipped_row,  # usecols=(0, 1)
                        )
    arraysT = np.transpose(arrays)  # NOQA

    # CONDITION 1: first column is the X-axis (e.g. time)
    t = arraysT[0]

    hide = []   # decide what channel to hide
    if plot_type == "channelCurrent":
        # hide = ['Nat', 'KAf', 'KDR', 'Cah', 'BK', 'SK']
        # hide = ['Nat', 'KAf', 'KDR', 'BK', 'SK']
        pass
    #############
    # Identify the array-index based on the defined time range
    idx_start = next(x[0] for x in enumerate(t) if x[1] >= time_start)
    if time_end == -1.0:
        idx_end = len(t)-1
        time_end = t[idx_end]
    else:
        idx_end = next(x[0] for x in enumerate(t) if x[1] >= time_end)
    # END
    #################
    arraysT = np.delete(arraysT, 0, 0)
    # arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
    # print arr
    # arr = np.delete(arr,0,1)
    # print arr
    # return
    # print t
    # print num_columns
    value_min = 200000000.0
    value_max = -value_min
    index_ii = 0
    if (plot_type in ["channelCurrent", "voltage", "voltagecalcium"]):
        for ydata in arraysT:
            # plot (t, col)
            #     legend(time, channel_names[i])
            color = COLORS[index_ii % len(COLORS)]
            style = LINE_STYLES[((index_ii / len(COLORS)) % len(LINE_STYLES))]
            # axarr[gr, gc].plot(t, ydata , color, label=channel_names[index_ii
            # % len(channel_names)])
            if channel_names[index_ii % len(channel_names)] not in hide:
                axes.plot(t, ydata, linestyle=style, color=color, label=channel_names[index_ii % len(channel_names)])
                axes.legend(ncol=2, prop={'size': 8}, loc=4)  # shadow=True, fancybox=True, loc="upper left", bbox_to_anchor=[0,1]
                # x = np.arange(0, 5, 0.1)
                # axes.plot(t, ydata, 'b-')
                value_min = min(np.amin(ydata[idx_start:idx_end]), value_min)
                value_max = max(np.amax(ydata[idx_start:idx_end]), value_max)
            # print len(t), len(ydata)
            index_ii += 1
        # print i
        # print value_min, value_max
        axes.set_ylim(bottom=value_min - 0.05 * max(0.1, abs(value_min)))
        axes.set_ylim(top=value_max + 0.05 * max(0.1, abs(value_max)))
        axes.set_xlim(left=time_start, right=time_end)

    if plot_type == "channelCurrent":
        axes.set_title("time (ms) vs. currents (pA/um^2)")
    elif plot_type == "voltagecalcium":
        axes.set_title("time (ms) vs. Vm | Calcium")

    # duplicated axis
    if plot_type == "voltagecalcium":
        channel_names = ["Ca"]
        num_columns = len(channel_names)  # NOQA

        arrays = np.loadtxt(my_file2,
                            skiprows=2,
                            # usecols=(0, 1)
                            )
        arraysT = np.transpose(arrays)  # NOQA
        t = arraysT[0]
        #############
        # Define time range
        idx_start = next(x[0] for x in enumerate(t) if x[1] >= time_start)
        if time_end == -1.0:
            idx_end = len(t)-1
            time_end = t[idx_end]
        else:
            try:
                idx_end = next(x[0] for x in enumerate(t) if x[1] >= time_end)
            except StopIteration as e:  # NOQA
                idx_end = len(t)-1
        # END
        #################
        arraysT = np.delete(arraysT, 0, 0)
        # arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
        # print arr
        # arr = np.delete(arr,0,1)
        # print arr
        # return
        # print t
        # print num_columns
        value_min = 200000000.0
        value_max = -value_min
        index_ii = 0
        # gr = 0; gc = 1 # graph row, col
        ax2 = axes.twinx()
        for ydata in arraysT:
            # plot (t, col)
            #     legend(time, channel_names[i])
            color = COLORS[(index_ii + 1) % len(COLORS)]
            style = LINE_STYLES[((index_ii / len(COLORS)) % len(LINE_STYLES))]
            ax2.plot(t, ydata, linestyle=style, color=color, label=channel_names[index_ii % len(channel_names)])
            ax2.legend(loc=0)
            value_min = min(np.amin(ydata[idx_start:idx_end]), value_min)
            value_max = max(np.amax(ydata[idx_start:idx_end]), value_max)
            # print len(t), len(ydata)
            index_ii += 1
        # s2 = np.sin(2*np.pi*t)
        # ax2.plot(t, s2, 'r.')
        # ax2.set_ylabel('sin', color='r')
        ax2.set_ylim(bottom=value_min - 0.05 * max(0.1, abs(value_min)))
        ax2.set_ylim(top=value_max + 0.05 * max(0.1, abs(value_max)))
        ax2.set_xlim(left=time_start, right=time_end)
        for tl in ax2.get_yticklabels():
            tl.set_color(color)
        # print value_min, value_max
    # other type of data
    # ...

    #
    axes.set_ylabel(label_yaxis)
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()
    return


def write_pdf_new_page_calcium_dye(pdf_file_handlers,
                                   my_file, pdf_filename,
                                   fluorescence_total, Kd,
                                   basal_calcium,
                                   time_start, time_end, label_yaxis,  # pylint: disable=too-complex
                                   plot_type, width_inch=3, height_inch=3,
                                   keep_holding_figure=False, continue_figure=False,
                                   custom_labels=None, offset_xaxis=None,
                                   hide_channels=[],
                                   plot_columns=[],
                                   plot_all=False):  # pylint: disable=too-complex
    """
    my_file = data file (column-based data; with 2 or many columns)
              the first column is typically time-information
              (but can also something else)
    pdf_filename = string name of pdf file
    fluorescence_total = [uM]
    Kd  = [uM] - dissociation constant
    time_start
    time_end    = range of data to plot
                (can be time or something else)
                [assume the first column in data-file is this information]
    label_yaxis     = label o Y-axis
    plot_type   = the type of data [help to customize the data-read]
    width_inch
    height_inch      = height in inches
    keep_holding_figure = if True, then wait for another data to be plotted
        it will use: _storedFig, _storedAxes, _offset
        to draw new lines on the existing plot
    continue_figure   = if True, then use the Figure stored in _storedFig, _storedAxes, _offset
    custom_labels      = user-defined label, i.e. not reading from first-line header
                    (use for data from the second-column beyond)
    hide_channels = list of channel names that are skipped (only for plot_type=="channelCurrent")
    plot_columns = if [], then this parameter has no role
                   if not empty, then this parameter indicate the column-indices of data to be plotted
                   (when plot_all==True)
    plot_all     = if False, then this parameter has no role
                   if True, then it performs plotting all columns (with an option to limit
                   what column indices using plot_columns option)
    NOTE: The axes range are based on the data on first file if
              multiple files are used
    """
    _acceptedPlotType = ["channelCurrent", "voltage", "concentrationCalcium"]
    if plot_type not in _acceptedPlotType:
        logging.error("Not valid plot_type")
        logging.error("... using " + plot_type)
        logging.error("...accepted: " + ', '.join(_acceptedPlotType))
        return

    if pdf_filename not in pdf_file_handlers:
        pdf = PdfPages(pdf_filename)
        pdf_file_handlers[pdf_filename] = pdf
    else:
        pdf = pdf_file_handlers[pdf_filename]

    fig = None
    axes = None
    if continue_figure is False:
        fig = plt.figure(figsize=(width_inch, height_inch))  # size of page 1 in inches
        axes = fig.add_subplot(111)
        globals.storedFig = fig
        globals.storedAxes = axes
        globals.offset = 0
    else:
        fig = globals.storedFig
        axes = globals.storedAxes
        globals.offset += 1

    if not os.path.isfile(my_file):
        logging.info("File %s not present", my_file)
        if keep_holding_figure is False:
            axes.set_ylabel(label_yaxis)
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()
        return

    num_columns = 1
    num_skipped_row = 1
    if plot_type == "channelCurrent":
        file = open(my_file)
        lines = file.readlines()
        # channel_names = re.split(",", lines[1])
        channel_names = re.split(",", re.sub(r'[\s+]', '', lines[1]))
        del channel_names[-1], channel_names[0]
        # print lines[1]
        num_skipped_row = 2
    elif plot_type == "voltage":
        channel_names = ["Vm"]
        if custom_labels is not None:
            if isinstance(custom_labels, list):
                channel_names = custom_labels
            else:
                logging.error("expect a list for 'custom_labels' argument")
                return
    elif plot_type == "concentrationCalcium":
        channel_names = ["Calcium"]
        if custom_labels is not None:
            if isinstance(custom_labels, list):
                channel_names = custom_labels
            else:
                logging.error("expect a list for 'custom_labels' argument")
                return

    print(channel_names)  # NOQA
    num_columns = len(channel_names)  # NOQA
    try:
        arrays = np.loadtxt(my_file,
                            skiprows=num_skipped_row,
                            # usecols=(0, 1)
                            )
    except ValueError:
        print("Error reading file %s with skip = %d rows" % (my_file, num_skipped_row))
        return
    arraysT = np.transpose(arrays)

    # CONDITION 1: first column is the X-axis (e.g. time)
    t = arraysT[0]
    if offset_xaxis is not None:
        t = t + offset_xaxis

    hide = []   # decide what channel to hide
    if plot_type == "channelCurrent":
        hide = hide_channels
        # hide = ['Nat', 'KAf', 'KDR', 'Cah', 'BK', 'SK']
        # hide = ['Nat', 'KAf', 'KDR', 'BK', 'SK']
        # pass
    #############
    # Identify the array-index based on the defined time range
    idx_start = next(x[0] for x in enumerate(t) if x[1] >= time_start)
    if time_end == -1.0:
        idx_end = len(t)-1
        time_end = t[idx_end]
    else:
        idx_end = next(x[0] for x in enumerate(t) if x[1] >= time_end)
    # END
    #################
    arraysT = np.delete(arraysT, 0, 0)
    # arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
    # print arr
    # arr = np.delete(arr,0,1)
    # print arr
    # return
    # print t
    # print num_columns
    value_min = 200000000.0
    value_max = -value_min
    index_ii = 0
    F0 = basal_calcium / (basal_calcium + Kd)
    if (plot_type in ["channelCurrent", "voltage", "concentrationCalcium"]):
        for ydata in arraysT:
            # plot (t, col)
            # legend(time, channel_names[i])
            ydata = ydata / (ydata + Kd) - F0
            color = COLORS[(index_ii + globals.offset) % len(COLORS)]
            style = LINE_STYLES[((index_ii / len(COLORS)) % len(LINE_STYLES))]
            # axarr[gr, gc].plot(t, ydata, color, label=channel_names[index_ii])
            if plot_all is True:
                if plot_columns:
                    if (index_ii + 2 in plot_columns):
                        axes.plot(t, ydata, linestyle=style,
                                  color=color, label=channel_names[index_ii % num_columns])
                        axes.legend(ncol=2, prop={'size': 8},
                                    loc=4, handlelength=3)  # shadow=True, fancybox=True, loc="upper left", bbox_to_anchor=[0,1]
                        value_min = min(np.amin(ydata[idx_start:idx_end]), value_min)
                        value_max = max(np.amax(ydata[idx_start:idx_end]), value_max)
                else:
                    axes.plot(t, ydata, linestyle=style,
                              color=color, label=channel_names[index_ii % num_columns])
                    axes.legend(ncol=2, prop={'size': 8},
                                loc=4, handlelength=3)  # shadow=True, fancybox=True, loc="upper left", bbox_to_anchor=[0,1]
                    value_min = min(np.amin(ydata[idx_start:idx_end]), value_min)
                    value_max = max(np.amax(ydata[idx_start:idx_end]), value_max)
                # NOTE: handlelength ensure proper dotted-line show in legend
                # x = np.arange(0, 5, 0.1)
                # axes.plot(t, ydata, 'b-')
                # value_min = min(np.amin(ydata[idx_start:idx_end]), value_min)
                # value_max = max(np.amax(ydata[idx_start:idx_end]), value_max)
            else:
                if channel_names[index_ii] not in hide:
                    axes.plot(t, ydata, linestyle=style, color=color, label=channel_names[index_ii])
                    axes.legend(ncol=2, prop={'size': 8}, loc=4, handlelength=3)  # shadow=True, fancybox=True, loc="upper left", bbox_to_anchor=[0,1]
                    # NOTE: handlelength ensure proper dotted-line show in legend
                    # x = np.arange(0, 5, 0.1)
                    # axes.plot(t, ydata, 'b-')
                    value_min = min(np.amin(ydata[idx_start:idx_end]), value_min)
                    value_max = max(np.amax(ydata[idx_start:idx_end]), value_max)
            # print len(t), len(ydata)
            index_ii += 1
        # print i
        # print value_min, value_max
        axes.set_ylim(bottom=value_min - 0.05 * max(0.1, abs(value_min)))
        axes.set_ylim(top=value_max + 0.05 * max(0.1, abs(value_max)))
        axes.set_xlim(left=time_start, right=time_end)
        if plot_type == "channelCurrent":
            axes.set_title("time (ms) vs. currents (pA/um^2)")
        if plot_type == "voltage":
            axes.set_title("time (ms) vs. voltage (mV)")

    # other type of data
    # ...

    #
    if keep_holding_figure is False:
        axes.set_ylabel(label_yaxis)
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()
    return


def get_list_indices(my_file, pattern):
    """return the array of indices"""
    # num_skipped_row = 2
    # arrays = np.loadtxt(my_file,
    #                     skiprows=num_skipped_row
    #                     # usecols=(0, 1)
    #                     )
    # arraysT = np.transpose(arrays)
    # arraysT = np.delete(arraysT, 0, 0)
    # numcols = len(arrays[0])
    list_indices = range(2, 10)
    return list_indices


def write_pdf_new_page_overlap_spines(pdf_file_handlers,
                                      my_file, my_file2,
                                      pdf_filename, time_start, time_end,
                                      label_yaxis, plot_type,
                                      list_indices,
                                      width_inch=3, height_inch=3):
    """
    Plot Vm + Ca together for data from many spines
      and we limit to those specified in list_indices (starting from 2)

    my_file = data file (column-based data; with 2 or many columns)
              the first column is typically time-information
              (but can also something else)
    my_file2 = data file (column-based data; with 2 or many columns)
              the first column is typically time-information
              (but can also something else)
    time_start
    time_end    = range of data to plot
                (can be time or something else)
                [assume the first column in data-file is this information]
    label_yaxis     = label o Y-axis
    plot_type   = the type of data [help to customize the data-read]
    width_inch
    height_inch      = height in inches
    """
    if not os.path.isfile(my_file):
        logging.info("File %s not present", my_file)
        return

    _acceptedPlotType = ["channelCurrent", "voltage", "concentrationCalcium",
                         "voltagecalcium"]
    if plot_type not in _acceptedPlotType:
        logging.info("Not valid plot_type")
        logging.info("... using " + plot_type)
        logging.info("...accepted " + _acceptedPlotType)
        return

    pdf = None
    if pdf_filename not in pdf_file_handlers:
        pdf = PdfPages(pdf_filename)
        pdf_file_handlers[pdf_filename] = pdf
    else:
        pdf = pdf_file_handlers[pdf_filename]
    fig = plt.figure(figsize=(width_inch, height_inch))  # size of page 1 in inches
    # axes = fig.add_subplot(111)
    axes = fig.gca()

    num_columns = 1
    num_skipped_row = 1
    if plot_type == "channelCurrent":
        file = open(my_file)
        lines = file.readlines()
        # channel_names = re.split(",", lines[1])
        channel_names = re.split(",", re.sub(r'[\s+]', '', lines[1]))
        del channel_names[-1], channel_names[0]
        # print lines[1]
        # print channel_names
        num_skipped_row = 2
    elif plot_type == "voltagecalcium":
        channel_names = ["Vm"]

    num_columns = len(channel_names)
    arrays = np.loadtxt(my_file,
                        skiprows=num_skipped_row  #, usecols=(0, 1)
                        )
    arraysT = np.transpose(arrays)  # NOQA
    # TUAN TODO : update reading CSV (and try to update the C++ writer)
    # arrays = pd.read_csv(my_file, skiprows=num_skipped_row)

    # CONDITION 1: first column is the X-axis (e.g. time)
    t = arraysT[0]

    hide = []   # decide what channel to hide
    if plot_type == "channelCurrent":
        # hide = ['Nat', 'KAf', 'KDR', 'Cah', 'BK', 'SK']
        # hide = ['Nat', 'KAf', 'KDR', 'BK', 'SK']
        pass
    #############
    # Identify the array-index based on the defined time range
    idx_start = next(x[0] for x in enumerate(t) if x[1] >= time_start)
    if time_end == -1.0:
        idx_end = len(t)-1
        time_end = t[idx_end]
    else:
        idx_end = next(x[0] for x in enumerate(t) if x[1] >= time_end)
    # END
    #################
    arraysT = np.delete(arraysT, 0, 0)
    # arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
    # print arr
    # arr = np.delete(arr,0,1)
    # print arr
    # return
    # print t
    # print num_columns
    value_min = 200000000.0
    value_max = -value_min
    index_ii = 0
    if (plot_type in ["channelCurrent", "voltage", "voltagecalcium"]):
        for ydata in arraysT:
            # NOTE: list_indices with minimal value is 2
            if index_ii+2 in list_indices:
                # plot (t, col)
                #     legend(time, channel_names[i])
                color = COLORS[index_ii % len(COLORS)]
                style = LINE_STYLES[((index_ii / len(COLORS)) % len(LINE_STYLES))]
                # axarr[gr, gc].plot(t, ydata , color, label=channel_names[index_ii
                # % len(channel_names)])
                if channel_names[index_ii % len(channel_names)] not in hide:
                    axes.plot(t, ydata, linestyle=style, color=color, label=channel_names[index_ii % len(channel_names)])
                    axes.legend(ncol=2, prop={'size': 8}, loc=4)  # shadow=True, fancybox=True, loc="upper left", bbox_to_anchor=[0,1]
                    # x = np.arange(0, 5, 0.1)
                    # axes.plot(t, ydata, 'b-')
                    value_min = min(np.amin(ydata[idx_start:idx_end]), value_min)
                    value_max = max(np.amax(ydata[idx_start:idx_end]), value_max)
                # print len(t), len(ydata)
            index_ii += 1
        # print i
        # print value_min, value_max
        axes.set_ylim(bottom=value_min - 0.05 * max(0.1, abs(value_min)))
        axes.set_ylim(top=value_max + 0.05 * max(0.1, abs(value_max)))
        axes.set_xlim(left=time_start, right=time_end)

    if plot_type == "channelCurrent":
        axes.set_title("time (ms) vs. currents (pA/um^2)")
    elif plot_type == "voltagecalcium":
        axes.set_title("time (ms) vs. Vm | Calcium")

    # duplicated axis
    if plot_type == "voltagecalcium":
        channel_names = ["Ca"]
        num_columns = len(channel_names)  # NOQA

        arrays = np.loadtxt(my_file2
                            , skiprows=2
                            # , usecols=(0, 1)
                            )
        arraysT = np.transpose(arrays)  # NOQA
        t = arraysT[0]
        #############
        # Define time range
        idx_start = next(x[0] for x in enumerate(t) if x[1] >= time_start)
        if time_end == -1.0:
            idx_end = len(t)-1
            time_end = t[idx_end]
        else:
            try:
                idx_end = next(x[0] for x in enumerate(t) if x[1] >= time_end)
            except StopIteration as e:  # NOQA
                idx_end = len(t)-1
        # END
        #################
        arraysT = np.delete(arraysT, 0, 0)
        # arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
        # print arr
        # arr = np.delete(arr,0,1)
        # print arr
        # return
        # print t
        # print num_columns
        value_min = 200000000.0
        value_max = -value_min
        index_ii = 0
        # gr = 0; gc = 1 # graph row, col
        ax2 = axes.twinx()
        for ydata in arraysT:
            # NOTE: list_indices with minimal value is 2
            if index_ii+2 in list_indices:
                # plot (t, col)
                #     legend(time, channel_names[i])
                color = COLORS[(index_ii + 1) % len(COLORS)]
                style = LINE_STYLES[((index_ii / len(COLORS)) % len(LINE_STYLES))]
                ax2.plot(t, ydata, linestyle=style, color=color, label=channel_names[index_ii % len(channel_names)])
                ax2.legend(loc=0)
                value_min = min(np.amin(ydata[idx_start:idx_end]), value_min)
                value_max = max(np.amax(ydata[idx_start:idx_end]), value_max)
                # print len(t), len(ydata)
            index_ii += 1
        # s2 = np.sin(2*np.pi*t)
        # ax2.plot(t, s2, 'r.')
        # ax2.set_ylabel('sin', color='r')
        ax2.set_ylim(bottom=value_min - 0.05 * max(0.1, abs(value_min)))
        ax2.set_ylim(top=value_max + 0.05 * max(0.1, abs(value_max)))
        ax2.set_xlim(left=time_start, right=time_end)
        for tl in ax2.get_yticklabels():
            tl.set_color(color)
        # print value_min, value_max
    # other type of data
    # ...

    #
    axes.set_ylabel(label_yaxis)
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()
    return


def write_pdf_new_page_spines(pdf_file_handlers,
                              my_file, pdf_filename,
                              time_start, time_end,
                              list_indices,
                              label_yaxis,
                              plot_type, width_inch=3, height_inch=3,
                              keep_holding_figure=False, continue_figure=False,
                              custom_labels=None, offset_xaxis=None,
                              hide_channels=[],
                              show_channels=[],
                              plot_columns=[],
                              plot_all=False,
                              convert_unit=["current", "pA/um^2"]):
    """
    my_file = data file (column-based data; with 2 or many columns)
              the first column is typically time-information
              (but can also something else)
    time_start
    time_end    = range of data to plot
                (can be time or something else)
                [assume the first column in data-file is this information]
    list_indices = [list of indices] of spine neck/head from that we can extract the currents data
                HERE, we assume all spines have the same types of currents
                otherwise, it is hard to break down what current belong to what spine
    label_yaxis     = label o Y-axis
    plot_type   = the type of data [help to customize the data-read]
    width_inch
    height_inch      = height in inches
    keep_holding_figure = if True, then wait for another data to be plotted
        it will use: _storedFig, _storedAxes, _offset
        to draw new lines on the existing plot
    continue_figure   = if True, then use the Figure stored in _storedFig, _storedAxes, _offset
    custom_labels      = user-defined label, i.e. not reading from first-line header
                    (use for data from the second-column beyond)
    hide_channels = list of channel names that are skipped (only for plot_type=="channelCurrent")
    show_channels = list of channel names that are plotted (only for plot_type=="channelCurrent")
                NOTE: hide_channels and show_channels are EXCLUSIVE
    plot_columns = if [], then this parameter has no role
                   if not empty, then this parameter indicate the column-indices of data to be plotted
                   (when plot_all==True)
    plot_all     = if False, then this parameter has no role
                   if True, then it performs plotting all columns (with an option to limit
                   what column indices using plot_columns option)
    convert_unit  = ["current", "pA/um^2"]  -- default
                    ["current", "uA/cm^2"]
                    ["current", "mA/cm^2"]
                    ["current", "nA", surface_area]  - whole-cell
                    [default is pA/um^2 for current-density]
                    NOTE: surface_area = [um^2]
    NOTE: The axes range are based on the data on first file if
              multiple files are used
    """
    def get_names(my_file, channel_names):
        num_columns = 1
        num_skipped_row = 1
        if plot_type == "channelCurrent":
            file = open(my_file)
            lines = file.readlines()
            # channel_names = re.split(",", lines[1])
            channel_names = re.split(",", re.sub(r'[\s+]', '', lines[1]))
            del channel_names[-1], channel_names[0]
            # print lines[1]
            num_skipped_row = 2
        elif plot_type == "voltage":
            channel_names = ["Vm"]
            if custom_labels is not None:
                if isinstance(custom_labels, list):
                    channel_names = custom_labels
                else:
                    logging.error("expect a list for 'custom_labels' argument")
                    return
        elif plot_type == "concentrationCalcium":
            channel_names = ["Calcium"]
            if custom_labels is not None:
                if isinstance(custom_labels, list):
                    channel_names = custom_labels
                else:
                    logging.error("expect a list for 'custom_labels' argument")
                    return
        elif plot_type == "currentStim":
            channel_names = ["Istim"]
            if custom_labels is not None:
                if isinstance(custom_labels, list):
                    channel_names = custom_labels
                else:
                    logging.error("expect a list for 'custom_labels' argument")
                    return
        return [num_columns, num_skipped_row]
    _acceptedPlotType = ["channelCurrent", "voltage", "concentrationCalcium", "currentStim"]
    if plot_type not in _acceptedPlotType:
        logging.error("Not valid plot_type")
        logging.error("... using " + plot_type)
        logging.error("...accepted: " + ', '.join(_acceptedPlotType))
        return

    if pdf_filename not in pdf_file_handlers:
        pdf = PdfPages(pdf_filename)
        pdf_file_handlers[pdf_filename] = pdf
    else:
        pdf = pdf_file_handlers[pdf_filename]

    fig = None
    axes = None
    if continue_figure is False:
        if os.path.isfile(my_file):  # if file NOT exist, do not create new PDf page
            # add another constraint before creating new PDF page
            channel_names = []
            [num_columns, num_skipped_row] = get_names(my_file, channel_names)
            if (show_channels and set(channel_names).isdisjoint(show_channels)) or \
                    (hide_channels and set(channel_names).issubset(hide_channels)):
                fig = plt.figure(figsize=(width_inch, height_inch))  # size of page 1 in inches
                axes = fig.add_subplot(111)
                globals.storedFig = fig
                globals.storedAxes = axes
                globals.offset = 0
    else:
        fig = globals.storedFig
        axes = globals.storedAxes
        globals.offset += 1

    if not os.path.isfile(my_file):
        logging.info("File %s not present", my_file)
        if keep_holding_figure is False and continue_figure is True:
            axes.set_ylabel(label_yaxis)
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()
        return

    channel_names = []
    [num_columns, num_skipped_row] = get_names(my_file, channel_names)

    # print(channel_names)  # NOQA
    num_columns = len(channel_names)  # NOQA
    try:
        arrays = np.loadtxt(my_file,
                            skiprows=num_skipped_row,
                            # usecols=(0, 1)
                            )
    except ValueError:
        print("Error reading file %s with skip = %d rows" % (my_file, num_skipped_row))
        return
    arraysT = np.transpose(arrays)

    # CONDITION 1: first column is the X-axis (e.g. time)
    t = arraysT[0]
    if offset_xaxis is not None:
        t = t + offset_xaxis

    hide = []   # decide what channel to hide
    if plot_type == "channelCurrent":
        hide = hide_channels
        # hide = ['Nat', 'KAf', 'KDR', 'Cah', 'BK', 'SK']
        # hide = ['Nat', 'KAf', 'KDR', 'BK', 'SK']
        # pass
    if hide_channels and show_channels:
        logging.error("Either hide_channels xor show_channels but not both")
        return
    #############
    # Identify the array-index based on the defined time range
    idx_start = next(x[0] for x in enumerate(t) if x[1] >= time_start)
    if time_end == -1.0:
        idx_end = len(t)-1
        time_end = t[idx_end]
    else:
        idx_end = next(x[0] for x in enumerate(t) if x[1] >= time_end)
    # END
    #################
    arraysT = np.delete(arraysT, 0, 0)
    # arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
    # print arr
    # arr = np.delete(arr,0,1)
    # print arr
    # return
    # print t
    # print num_columns
    value_min = 200000000.0
    value_max = -value_min
    index_ii = 0
    setChanName = set(channel_names)
    print(len(setChanName))
    quit()
    if (plot_type in ["channelCurrent", "voltage", "concentrationCalcium"]):
        for ydata in arraysT:
            if plot_type in ["channelCurrent"] and convert_unit[0:2] == ["current", "uA/cm^2"]:
                ydata = 100 * ydata  # [convert from pA/um^2 to uA/cm^2]
            elif plot_type in ["channelCurrent"] and convert_unit[0:2] == ["current", "mA/cm^2"]:
                ydata = 0.1 * ydata  # [convert from pA/um^2 to mA/cm^2]
            elif plot_type in ["channelCurrent"] and convert_unit[0:2] == ["current", "nA"]:
                ydata = ydata * convert_unit[2] * 1e-3  # [convert from pA/um^2 to nA]
            # plot (t, col)
            # legend(time, channel_names[i])
            color = COLORS[(index_ii + globals.offset) % len(COLORS)]
            style = LINE_STYLES[((index_ii / len(COLORS)) % len(LINE_STYLES))]
            # axarr[gr, gc].plot(t, ydata, color, label=channel_names[index_ii])
            if plot_all is True:
                if plot_columns:
                    if index_ii + 2 in plot_columns:
                        axes.plot(t, ydata, linestyle=style,
                                  color=color, label=channel_names[index_ii % num_columns])
                        axes.legend(ncol=2, prop={'size': 8},
                                    loc=4, handlelength=3)  # shadow=True, fancybox=True, loc="upper left", bbox_to_anchor=[0,1]
                        value_min = min(np.amin(ydata[idx_start:idx_end]), value_min)
                        value_max = max(np.amax(ydata[idx_start:idx_end]), value_max)
                else:
                    axes.plot(t, ydata, linestyle=style,
                              color=color, label=channel_names[index_ii % num_columns])
                    axes.legend(ncol=2, prop={'size': 8},
                                loc=4, handlelength=3)  # shadow=True, fancybox=True, loc="upper left", bbox_to_anchor=[0,1]
                    value_min = min(np.amin(ydata[idx_start:idx_end]), value_min)
                    value_max = max(np.amax(ydata[idx_start:idx_end]), value_max)
                # NOTE: handlelength ensure proper dotted-line show in legend
                # x = np.arange(0, 5, 0.1)
                # axes.plot(t, ydata, 'b-')
                # value_min = min(np.amin(ydata[idx_start:idx_end]), value_min)
                # value_max = max(np.amax(ydata[idx_start:idx_end]), value_max)
            else:
                if (show_channels and channel_names[index_ii] in show_channels) or \
                        (hide and channel_names[index_ii] not in hide) or \
                        (not hide and not show_channels):
                    axes.plot(t, ydata, linestyle=style, color=color, label=channel_names[index_ii])
                    axes.legend(ncol=2, prop={'size': 8}, loc=4, handlelength=3)  # shadow=True, fancybox=True, loc="upper left", bbox_to_anchor=[0,1]
                    # NOTE: handlelength ensure proper dotted-line show in legend
                    # x = np.arange(0, 5, 0.1)
                    # axes.plot(t, ydata, 'b-')
                    value_min = min(np.amin(ydata[idx_start:idx_end]), value_min)
                    value_max = max(np.amax(ydata[idx_start:idx_end]), value_max)
            # print len(t), len(ydata)
            index_ii += 1
        # print i
        # print value_min, value_max
        # axes.set_ylim(bottom=value_min - 0.05 * max(0.1, abs(value_min)))
        # axes.set_ylim(top=value_max + 0.05 * max(0.1, abs(value_max)))
        axes.set_ylim(bottom=value_min - 0.05 * abs(value_min))
        axes.set_ylim(top=value_max + 0.05 * abs(value_max))
        axes.set_xlim(left=time_start, right=time_end)
        if plot_type == "channelCurrent":
            if convert_unit[0:2] == ["current", "uA/cm^2"]:
                axes.set_title(r"time (ms) vs. currents ($\mu$A/cm$^2$)")
            elif convert_unit[0:2] == ["current", "mA/cm^2"]:
                axes.set_title(r"time (ms) vs. currents (pA/cm$^2$)")
            elif convert_unit[0:2] == ["current", "pA/um^2"]:
                axes.set_title(r"time (ms) vs. currents (pA/$\mu$m$^2$)")
            elif convert_unit[0:2] == ["current", "nA"]:
                axes.set_title(r"time (ms) vs. currents (nA)")
        if plot_type == "voltage":
            axes.set_title("time (ms) vs. voltage (mV)")

    # other type of data
    # ...
    if (plot_type in ["currentStim"]):
        for ydata in arraysT:
            color = COLORS[(index_ii+globals.offset) % len(COLORS)]
            style = LINE_STYLES[((index_ii / len(COLORS)) % len(LINE_STYLES))]
            axes.plot(t, ydata, linestyle=style, color=color, label=channel_names[index_ii])
            axes.legend(ncol=2, prop={'size': 8}, loc=4, handlelength=3)  # shadow=True, fancybox=True, loc="upper left", bbox_to_anchor=[0,1]
            # NOTE: handlelength ensure proper dotted-line show in legend
            # x = np.arange(0, 5, 0.1)
            # axes.plot(t, ydata, 'b-')
            value_min = min(np.amin(ydata[idx_start:idx_end]), value_min)
            value_max = max(np.amax(ydata[idx_start:idx_end]), value_max)
        axes.set_ylim(bottom=value_min - 0.05 * max(0.1, abs(value_min)))
        axes.set_ylim(top=value_max + 0.05 * max(0.1, abs(value_max)))
        axes.set_xlim(left=time_start, right=time_end)
        if plot_type == "currentStim":
            if convert_unit[0:2] == ["current", "pA"]:
                axes.set_title(r"time (ms) vs. currents (pA)")
    #
    if keep_holding_figure is False:
        axes.set_ylabel(label_yaxis)
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()
    return
