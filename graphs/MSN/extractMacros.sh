#!/bin/bash
#author  = Hoang Trong Minh Tuan (@IBM - 2016)
#version = 1.0

#{{{USER-SETTING
_X_=1
_Y_=1
_Z_=1
NUMTHREADS=1

NUMPROCESSES=$(( _X_ * _Y_ * _Z_))
OUTPUTFOLDER=`echo $HOME`/NTS_OUTPUT/
if [ ! -d ${OUTPUTFOLDER} ]; then  mkdir ${OUTPUTFOLDER}; fi
#}}}

numArgs=$#
secondArg=$2

#{{{ Non-modified parts
#Escape code
esc=`echo -en "\033"`

# Set colors
cc_red="${esc}[0;31m"
cc_green="${esc}[0;32m"
cc_yellow="${esc}[0;33m"
cc_blue="${esc}[0;34m"
cc_normal=`echo -en "${esc}[m\017"`

Yes_No_RunSim()
{
  #{{{
  # print question
  echo -n "The folder exist; want to override ?(yes(y)/no(n)): "

  # read answer
  read YnAnswer

  # all to lower case
  YnAnswer=$(echo $YnAnswer | awk '{print tolower($0)}')

  # check and act on given answer
  case $YnAnswer in
    "yes")  RunSim;;
    "y")  RunSim;;
    "no")  ;;
    "n")  ;;
    *)      echo "Please answer yes(y) or no(n)" ; Yes_No_RunSim;;
  esac
  #}}}
}

RunSim()
{
  #{{{
   echo "Output Folder: " $OutputFolderName
   cp params $OutputFolderName/ -L -r
   cp *.gsl $OutputFolderName/ -L -r
   cp neurons.txt $OutputFolderName/ -L -r
   cp neurons/neuron.swc $OutputFolderName/ -L -r
   cp ../../gsl/bin/gslparser $OutputFolderName/ -L -r
   cp $NTSROOT/nti/include/Model2Use.h $OutputFolderName/ -L -r
   cp $NTSROOT/nti/include/NTSMacros.h  $OutputFolderName/ -L -r
   cp $NTSROOT/nti/include/MaxComputeOrder.h $OutputFolderName/ -L -r
   #cp spines $OutputFolderName/ -L -r
   SWC_FILENAME=`readlink -f ./neurons/neuron.swc`
   echo "----> $OutputFolderName" >> SIM_LOG
   echo "----> RESULT: " >> SIM_LOG
   echo "... using swc file: ${SWC_FILENAME}"
   echo "... using swc file: ${SWC_FILENAME}" >> SIM_LOG
   echo ./NTS_plotting/doPlot.sh  ${OUTPUTFOLDER} ${runCaseNumber} ${uniqueName:1} >> SIM_LOG
   echo "---------------------- " >> SIM_LOG
   cp SIM_LOG $OutputFolderName/ -L -r
   echo "Output Folder: " $OutputFolderName
   echo "GSL file: " $temp_file
   cp Topology.h .Topology_backup.h
   echo "#define _X_ $_X_" > Topology.h
   echo "#define _Y_ $_Y_" >> Topology.h
   echo "#define _Z_ $_Z_" >> Topology.h
   mpiexec -n ${NUMPROCESSES}  ../../gsl/bin/gslparser $temp_file -t ${NUMTHREADS}
   echo "Output Folder: " $OutputFolderName
   ## NOTE: comment out if we don't want to plot
   if [ $numArgs -eq 1 ] || [ "$secondArg" != "-noplot" ]; then
     cd NTS_plotting;./doPlot.sh  ${OUTPUTFOLDER} ${runCaseNumber} ${uniqueName:1}; cd -
   fi
   echo ./doPlot.sh  ${OUTPUTFOLDER} ${runCaseNumber} ${uniqueName:1} 
  #}}}
}

DoFinish()
{
  #{{{
  ##NOTE: each line in this file will be read in by Plot code
  fileListFolders=./.listFolders2Plot
  if [ ! -f $fileListFolders ]; then
    touch $fileListFolders
  fi
   echo "$OutputFolderName" >> $fileListFolders
  #}}}
}
#///}}}

#########################
## PROCESS
#########################
#########################
#{{{ 1. CREATE FILE/FOLDER
TMPDIR=`pwd`/.tmp
if [ ! -d $TMPDIR ]; then
  mkdir $TMPDIR
fi
FILENAME_PREVIOUSRUN=$TMPDIR/previousRun
if [ ! -f $FILENAME_PREVIOUSRUN ]; then
  touch $FILENAME_PREVIOUSRUN
fi
temp_file=`mktemp --tmpdir=$TMPDIR`
#}}}

##{{{ 2. CHECK ARGS
if [ "$#" == "0" ]; then
  echo -e "${cc_red}IMPORTANT${cc_normal}:  When using this script, you can modify the session 'USER-SETTING'
which include 1. num-processes, 2. num-threads; 3. location of output (OUTPUTFOLDER)
       "
  echo " Data output folder (a subfolder inside \${OUTPUTFOLDER}): "
  echo -e "     ${cc_blue}\${morph}${cc_red}\${model_specific}${cc_blue}\${EXTENSION}${cc_normal}"
  echo ""
  echo " \${morph}          value is defined inside model.gsl. E.g.: 'msn_'"
  echo " \${model_specific} value is defined inside model.gsl, and is named with 2 parts
            1. authorName
            2. simulation condition
          E.g. Tuan_rest   or Tuan_triggersoma"
  echo " \${EXTENSION} is a value defined based on the choice of <extension> to $0"
  echo ""
  echo "SYNTAX:"
  echo -e "${cc_blue} $0 <extension> [-noplot] ${cc_normal}"
  echo "OPTIONS:"
  echo " <extension> somename to make the output folder unique"
  echo " <extension>: "
  echo "     -unique    : then the script generate a unique name using 'date +'%Y-%m-%d-%s'' which evoke the date"
  echo "     -reuse     : then the simulation overwrite the folder previously used"
  echo "     abc        : then the simulation assign 'abc' to \${EXTENSION}"
  echo "Example: "
  echo " $0 -unique"
  echo " $0 -reuse"
  echo " $0 abc"
  echo "When -noplot is used; then no plotting is called after the simulation"
  echo ""
  echo "NOTE: GSL specific"
  echo " \${dataFolder}   : DEFAULT is './data' [can be modified if you use this script]"
  echo " \${paramFolder}  : DEFAULT is './params' [can be modified if you use this script]"

  exit
fi
if [ "$1" == "-unique" ]; then
  uniqueName=-`date +'%Y-%m-%d-%s'`
  echo $uniqueName > $FILENAME_PREVIOUSRUN
elif [ "$1" == "-reuse" ]; then
  line=$(head -n 1 $FILENAME_PREVIOUSRUN)
  if [ -z $line  ]; then
    echo "No previous runs yet !"
    exit
  else
    uniqueName=$line
  fi
else
  uniqueName=-$1
fi
#}}}

#########################
##{{{ 3. CHECK MACROS
## NOTE: accepted macros
## -DOUTPUTFOLDER=location where all output should be'
## -DPARAMFOLDER=location of the parameters to the simulation
## -DEXTENSION=a prefix that helps to make unique output folder
#Simplest-one: 
# cpp -dU -P model.gsl -DEXTENSION=$uniqueName > ${temp_file} 2> /dev/null
cpp -dU -P model.gsl -DEXTENSION=$uniqueName -DOUTPUTFOLDER="${OUTPUTFOLDER}" > ${temp_file} 2> /dev/null
###cpp -dU -P model.gsl -DEXTENSION=$uniqueName > out.txt 2> /dev/null
##awk -F '/^#define[[:space:]]+morph/{ printf "%s | %s \n", $2, $3 }' < ${temp_file}
#morph=`awk  '/^#define morph/{printf "%s\n", $3}' < ${temp_file}`
morph=`awk  '/^#define morph/{print $3}' < ${temp_file}`
#dataFolder=`awk  '/^#define dataFolder/{printf "%s\n", $3}' < ${temp_file}`
#dataFolder=`awk  '/^#define dataFolder/{print $3}' < ${temp_file}`
dataFolder=`awk  '/^#define OUTPUTFOLDER/{print $3}' < ${temp_file}`
suffix=`awk  '/^#define OutputFolderName/{ printf "%s \n", $5 }' < ${temp_file}`
runCaseName=`awk  '/^#define STIMULUS_CASE/{ print $3 }' < ${temp_file}`
runCaseNumber=`awk  '/^#define '${runCaseName}'/{ print $3 }' < ${temp_file}`
OutputFolderName=`echo $dataFolder | sed -e 's/^"//' -e 's/"$//'`
OutputFolderName+=`echo $morph | sed -e 's/^"//' -e 's/"$//'`
OutputFolderName+=`echo $suffix | sed -e 's/^"//' -e 's/"$//'`
OutputFolderName+=$uniqueName
##echo "$morph" | sed -e 's/^"//' -e 's/"$//'
#echo "morph="$morph
#echo $dataFolder
#echo $suffix
mv $OutputFolderName $TMPDIR/
#}}}

###########################
#{{{ 4. Final step
if [ ! -d $OutputFolderName ]; then
  mkdir $OutputFolderName
  RunSim
else
  Yes_No_RunSim
fi

DoFinish
rm ${temp_file}
#}}}
