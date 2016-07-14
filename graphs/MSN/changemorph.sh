#!/bin/bash
####
# This script was written to enable quicly switching from one morphology to another
# author: Tuan M. Hoang Trong (IBM, @2016)
#
if [ "$#" == "0" ]; then
  echo "$0 <prefix> "
  echo "    <prefix> is the prefix of the folder where (1) morphology, (2) params, (3) recording/stimulus sites for that morph are stored"
  echo "  e.g. msn0, unknown"
  echo "NOTE: $0 clean "
  echo "   to clean the symbolic links"
fi
if [ "$1" == "clean" ]; then
rm neurons
rm params
rm connect_recording_model.gsl
rm recording_model.gsl
rm connect_stimulus_model.gsl
rm stimulus_model.gsl
else
#NOTE: $1  = suffix (e.g. hay2, hay1, unknown)
rm neurons
ln -s neurons_$1 neurons
rm params
ln -s params_$1 params
rm connect_recording_model.gsl
ln -s neurons/connect_recording_model_$1.gsl connect_recording_model.gsl
rm recording_model.gsl
ln -s neurons/recording_model_$1.gsl recording_model.gsl
rm connect_stimulus_model.gsl
ln -s neurons/connect_stimulus_model_$1.gsl  connect_stimulus_model.gsl
rm stimulus_model.gsl
ln -s neurons/stimulus_model_$1.gsl stimulus_model.gsl
fi
