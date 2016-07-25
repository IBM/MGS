#!/bin/bash
####
# This script was written to enable quicly switching from one morphology to another
#  and one model to another
# author: Tuan M. Hoang Trong (IBM, @2016)
#
rm_if_link(){ [ ! -L "$1" ] || rm -v "$1"; }
clean_all() {
  rm_if_link neurons
  rm_if_link params
  rm_if_link connect_recording_model.gsl
  rm_if_link recording_model.gsl
  rm_if_link connect_stimulus_model.gsl
  rm_if_link stimulus_model.gsl
  rm_if_link model.gsl
}

ModelFolder=systems

if [ "$#" == "0" ]; then
  echo "$0 <morph_suffix> "
  echo "    <suffix> is the suffix of the folder where (1) morphology, (2) params, (3) recording/stimulus sites for that morph are stored"
  echo "  e.g. traub51, traub3, traub1"
  echo "NOTE: $0 clean "
  echo "   to clean the symbolic links"
  exit
fi
if [ "$1" == "clean" ]; then
  clean_all
else
  #NOTE: $1  = suffix (e.g. hay2, hay1, unknown)
  clean_all
  ln -s neurons_$1 neurons

  ln -s neurons_$1/params params
  ln -s neurons/connect_recording_model_$1.gsl connect_recording_model.gsl
  ln -s neurons/recording_model_$1.gsl recording_model.gsl
  ln -s neurons/connect_stimulus_model_$1.gsl  connect_stimulus_model.gsl
  ln -s neurons/stimulus_model_$1.gsl stimulus_model.gsl
fi
