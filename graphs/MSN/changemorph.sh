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
  #rm_if_link connect_recording_model.gsl
  rm_if_link recording_model.gsl
  #rm_if_link connect_stimulus_model.gsl
  rm_if_link stimulus_model.gsl
  rm_if_link model.gsl
  rm_if_link neurons.txt 
  rm_if_link spines
}

Yes_No_ChangeSWC ()
{
  # print question
  echo -n "Want to change SWC link ?(yes(y)/no(n)): "

  # read answer
  read YnAnswer

  # all to lower case
  YnAnswer=$(echo $YnAnswer | awk '{print tolower($0)}')

  # check and act on given answer
  case $YnAnswer in
    "yes")  ChangeSWC;;
    "y")  ChangeSWC;;
    "no")  ;;
    "n")  ;;
    *)      echo "Please answer yes(y) or no(n)" ; Yes_No_ChangeSWC ;;
  esac
}
ChangeSWC ()
{
  paramFold=($(find neurons/ -maxdepth 1 -type f -name '*.swc' ! -name 'neuron.swc' ! -name '*developed.swc' -printf "%f\n"))
  echo "SUGGEST: neurons.swc_tufted.swc_reviseRadius.swc for hay1"
  echo "Select one of this:"
  arrSize=${#paramFold[@]}
  for i in "${!paramFold[@]}"; do 
    printf "%s\t%s\n" "$i" "${paramFold[$i]}"
  done
  re='^[0-9]+$'
  while true; do
      read -p "Type in the number? " REPLY
      if  [[ $REPLY =~ $re ]] ; then
        if [ $REPLY -ge 0 ] && [ $REPLY -lt $arrSize ]; then 
          arg2="${paramFold[$REPLY]}"
          break; 
        fi;
      fi
  done
  rm_if_link neurons/neuron.swc 2>&1 >/dev/null
  cd neurons;ln -s $arg2 neuron.swc; cd -
  spineFolder=$(echo $swcFile | cut -f 1 -d '.')"_spines"
  extMorph=$(echo $swcFile | cut -f 1 -d '.')
  echo "Spines folder:  neurons/$spineFolder"
  if [ -d neurons/"$spineFolder" ]; then
    rm_if_link spines 2>&1 >/dev/null
    ln -s neurons/$spineFolder spines
    rm_if_link neurons.txt
    ln -s spines/neurons.txt neurons.txt
    rm_if_link connect_recording_model.gsl
    ln -s neurons/connect_recording_model_$extMorph.gsl connect_recording_model.gsl
    rm_if_link connect_stimulus_model.gsl
    ln -s neurons/connect_stimulus_model_$extMorph.gsl  connect_stimulus_model.gsl
  else 
    echo "$spineFolder not found"
    rm_if_link neurons.txt
    ln -s single_neuron.txt neurons.txt
  fi
}

ModelFolder=systems
authorName=""

if [ "$#" == "0" ]; then
  echo "$0 <morph_suffix> [author_suffix] "
  echo "    <morph_suffix> is the suffix of the folder where (1) morphology, (2) params, (3) recording/stimulus sites for that morph are stored"
  echo "  e.g. msn0, unknown"
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

  if [ "$#" == "2" ]; then
    ln -s neurons_$1/params_$2 params
    ln -s $ModelFolder/model_$2.gsl model.gsl
    authorName=$2
  else
    #paramFold = ($(find -maxdepth 1 -type d -name 'params_*'))
    paramFold=($(find neurons_$1 -maxdepth 1 -type d -name 'params_*'))
    echo "Select one of this:"
    arrSize=${#paramFold[@]}
    for i in "${!paramFold[@]}"; do 
      printf "%s\t%s\n" "$i" "${paramFold[$i]}"
    done
    #for item in ${paramFold[*]}
    #do
    #  printf "   %s;" $item
    #done
    re='^[0-9]+$'
    while true; do
        read -p "Type in the number? " REPLY
        if  [[ $REPLY =~ $re ]] ; then
          if [ $REPLY -ge 0 ] && [ $REPLY -lt $arrSize ]; then 
            arg2="${paramFold[$REPLY]}"
            break; 
          fi;
        fi
    done
    ln -s $arg2 params
    authorName=`echo $arg2 | cut -d'_' -f 3`
    ln -s $ModelFolder/model_$authorName.gsl model.gsl
  fi
  #ln -s neurons/connect_recording_model_$1.gsl connect_recording_model.gsl
  ln -s neurons/recording_model_$1.gsl recording_model.gsl
  #ln -s neurons/connect_stimulus_model_$1.gsl  connect_stimulus_model.gsl
  ln -s neurons/stimulus_model_$1.gsl stimulus_model.gsl
  ln -s neurons/neurons_$authorName.txt neurons.txt
  
  ls neurons/neuron.swc 2>&1 >/dev/null
  swcFile=""
  if [ $? != 0 ]; then 
    swcFile=""
  else
    swcFile=($(ls -l neurons/neuron.swc 2>/dev/null | awk '{print $11}' 2>/dev/null))
  fi
  echo "Please make sure neurons/neuron.swc link to the right file"
  if [ "$swcFile" != "" ]; then
    echo "Currently link to $swcFile"
    Yes_No_ChangeSWC
  else
      ChangeSWC
  fi


fi
