# import logging
# import re
# import btmorph
# import math

# import numpy as np
# import os.path
# import csv
# import fnmatch
from file_utils import write_pdf_new_page, write_pdf_new_page_overlap # NOQA
from file_utils import write_pdf_new_page_currents  # NOQA
from file_utils import write_pdf_new_page_calcium_dye  # NOQA
from file_utils import create_new_pdf_file, get_file, close_pdf_file  # NOQA
from file_utils import get_working_folder, get_pdf_folder  # NOQA
from file_utils import write_pdf_new_page_spines  # NOQA


def plotcurrent_in_group_multiple_compartments_one_branch(pdf_file_handlers, folder, filename,
                                                          label_yaxis,
                                                          pdf_filename,
                                                          time_start, time_end):
    """
    currents are plotted using the same set of plots, so we group them here
    """
    ##########################################################
    # Page 1
    ############################
    my_file = folder+'/'+get_file(folder, filename)
    # label_yaxis = "soma"
    plot_type = "channelCurrent"
    width_inch = 10
    height_inch = 3
    write_pdf_new_page_currents(
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
    my_file = folder+'/'+get_file(folder, filename)
    # label_yaxis = "soma"
    plot_type = "channelCurrent"
    width_inch = 10
    height_inch = 3
    show_channels = ["Nat", "KAf", "KAs"]
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
    # Page 1b
    ############################
    my_file = folder+'/'+get_file(folder, filename)
    # label_yaxis = "soma"
    plot_type = "channelCurrent"
    width_inch = 10
    height_inch = 3
    hide_channels = ["Nat", "KAf", "KAs"]
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
        hide_channels=hide_channels)
    ##########################################################
    # Page 1c
    ############################
    my_file = folder+'/'+get_file(folder, filename)
    # label_yaxis = "soma"
    plot_type = "channelCurrent"
    width_inch = 10
    height_inch = 3
    show_channels = ["NCX"]
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


def plotcurrent_in_group(pdf_file_handlers, folder, filename,
                         label_yaxis,
                         pdf_filename,
                         time_start, time_end):
    """
    currents are plotted using the same set of plots, so we group them here
    """
    # convert_unit = ["current", "uA/cm^2"]
    convert_unit = ["current", "mA/cm^2"]
    # convert_unit = ["current", "pA/um^2"]
    ##########################################################
    # Page 1 - plot all currents
    ############################
    my_file = folder+'/'+get_file(folder, filename)
    # label_yaxis = "soma"
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


def plotcurrent_in_group_spines(pdf_file_handlers, folder, filename,
                                label_yaxis,
                                pdf_filename,
                                time_start, time_end,
                                list_indices
                                ):
    """
    currents are plotted using the same set of plots, so we group them here
    """
    # convert_unit = ["current", "uA/cm^2"]
    convert_unit = ["current", "mA/cm^2"]
    # convert_unit = ["current", "pA/um^2"]
    ##########################################################
    # Page 1 - plot all currents
    ############################
    my_file = folder+'/'+get_file(folder, filename)
    # label_yaxis = "soma"
    plot_type = "channelCurrent"
    width_inch = 10
    height_inch = 3
    write_pdf_new_page_spines(
        pdf_file_handlers,
        my_file,
        pdf_filename,
        time_start,
        time_end,
        list_indices,
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
    write_pdf_new_page_spines(
        pdf_file_handlers,
        my_file,
        pdf_filename,
        time_start,
        time_end,
        list_indices,
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
    write_pdf_new_page_spines(
        pdf_file_handlers,
        my_file,
        pdf_filename,
        time_start,
        time_end,
        list_indices,
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
    write_pdf_new_page_spines(
        pdf_file_handlers,
        my_file,
        pdf_filename,
        time_start,
        time_end,
        list_indices,
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
    write_pdf_new_page_spines(
        pdf_file_handlers,
        my_file,
        pdf_filename,
        time_start,
        time_end,
        list_indices,
        label_yaxis,
        plot_type,
        width_inch=width_inch,
        height_inch=height_inch,
        show_channels=show_channels,
        convert_unit=convert_unit)
