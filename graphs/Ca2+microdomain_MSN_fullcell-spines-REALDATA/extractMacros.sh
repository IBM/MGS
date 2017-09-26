#!/bin/bash
#author  = Hoang Trong Minh Tuan (@IBM - 2016)
#version = 1.0

#{{{USER-SETTING
PLOT_FOLDER=./NTS_plotting
_X_=2
_Y_=2
_Z_=1
NUMTHREADS=8

NUMPROCESSES=$(( _X_ * _Y_ * _Z_))
OUTPUTFOLDER=`echo $HOME`/NTS_OUTPUT/
if [ ! -d ${OUTPUTFOLDER} ]; then  mkdir ${OUTPUTFOLDER}; fi


DoPlot()
{
  ## Modify this function to do plotting for the output data
  if [ "$RUNSIM_COMPLETED" == 1 ]; then
    ## NOTE: comment out if we don't want to plot
    if [ -d $PLOT_FOLDER ]; then 
      if [ $numArgs -eq 1 ] || [ "$secondArg" != "-noplot" ]; then
        cd ${PLOT_FOLDER}/ && ./doPlot.sh  ${OUTPUTFOLDER} ${runCaseNumber} ${uniqueName:1}  ${morph}
      fi
      echo ${PLOT_FOLDER}/doPlot.sh  ${OUTPUTFOLDER} ${runCaseNumber} ${uniqueName:1}  ${morph}
    else 
      echo $PLOT_FOLDER "not exist"
    fi

    if [ -f  "$OutputFolderName/somaV.dat0" ]; then
      xmgrace -nxy  $OutputFolderName/somaV.dat0 
    fi
    # if [ -f  "$OutputFolderName/SEClamp.txt" ]; then
    #   xmgrace -block $OutputFolderName/SEClamp.txt  -bxy 1:2 & 
    # fi
    python /data/tmhoangt/Scripts/NTS_plotting/plot_soma.py  $OutputFolderName/ &
    python /data/tmhoangt/Scripts/NTS_plotting/plot_ais.py  $OutputFolderName/ 
    #python /data/tmhoangt/Scripts/NTS_plotting/plot_spine.py --folder $OutputFolderName/ --prefix proximal_shaft &
    #python /data/tmhoangt/Scripts/NTS_plotting/plot_spine.py --folder $OutputFolderName/ --prefix proximal_spine &
    #python /data/tmhoangt/Scripts/NTS_plotting/plot_spine.py --folder $OutputFolderName/ --prefix distal_spine&
    #python /data/tmhoangt/Scripts/NTS_plotting/plot_spine.py --folder $OutputFolderName/ --prefix distal_shaft &
  fi
}
#}}}

numArgs=$#
secondArg=$2
thirdArg=$3
COMMENT=""

#{{{ Non-modified parts
RUNSIM_COMPLETED=0
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
#GSLPARSER=gslparser_regular
GSLPARSER=../../gsl/bin/gslparser
  #{{{
  echo "Output Folder: " $OutputFolderName
  cp params $OutputFolderName/ -L -r
  cp *.gsl $OutputFolderName/ -L -r
  cp neurons.txt $OutputFolderName/ -L -r
  cp neurons/neuron.swc $OutputFolderName/ -L -r
  cp ${GSLPARSER}  $OutputFolderName/ -L -r
  cp $NTSROOT/nti/include/Model2Use.h $OutputFolderName/ -L -r
  cp $NTSROOT/nti/include/NTSMacros.h  $OutputFolderName/ -L -r
  cp $NTSROOT/nti/include/MaxComputeOrder.h $OutputFolderName/ -L -r
  #cp spines $OutputFolderName/ -L -r
  SWC_FILENAME=`readlink -f ./neurons/neuron.swc`
  echo "----> $OutputFolderName" >> SIM_LOG
  echo "----> RESULT: " >> SIM_LOG
  echo "... using swc file: ${SWC_FILENAME}"
  echo "... using swc file: ${SWC_FILENAME}" >> SIM_LOG
  echo ./doPlot.sh  ${OUTPUTFOLDER} ${runCaseNumber} ${uniqueName:1} ${morph} >> SIM_LOG
  echo ${COMMENT} >> SIM_LOG
  echo "---------------------- " >> SIM_LOG
  cp SIM_LOG $OutputFolderName/ -L -r
  echo "Output Folder: " $OutputFolderName
  echo "GSL file: " $temp_file
  mv Topology.h .Topology_backup.h
  echo "#define _X_ $_X_" > Topology.h
  echo "#define _Y_ $_Y_" >> Topology.h
  echo "#define _Z_ $_Z_" >> Topology.h
  mpiexec -n ${NUMPROCESSES}  ${GSLPARSER} $temp_file -t ${NUMTHREADS}
  #mv .Topology_backup.h Topology.h
  echo "Output Folder: " $OutputFolderName
  echo "mpiexec -n ${NUMPROCESSES}  ${GSLPARSER} $temp_file -t ${NUMTHREADS}"
  RUNSIM_COMPLETED=1
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
## Detect options 
#{{{
PLATFORM="LINUX"
WHAT_TO_BUILD="all"
CLEAN=""
REBUILD=""
MCO=""
NTI_BUILD_MODE="debug=yes"
GSL_BUILD_MODE=""
optspec=":c:p:sr-:j:"
#/while getopts "$optspec" optchar; do
#/  case $optchar in
#/    p)
#/      echo "Platform: $OPTARG" >&2
#/      PLATFORM=$OPTARG
#/      ;;
#/    c)
#/      echo "Build: $OPTARG"
#/      WHAT_TO_BUILD=$OPTARG
#/      ;;
#/    j)
#/      echo "Num-processes: $OPTARG"
#/      NPROCS=$OPTARG
#/      ;;
#/    s)
#/      NPROCS=1
#/      ;;
#/    r) # release
#/      NTI_BUILD_MODE=""
#/      GSL_BUILD_MODE="--release"
#/      ;;
#/    -)  #parse long-option
#/            case "${OPTARG}" in
#/                unique)
#/                    uniqueName=-`date +'%Y-%m-%d-%s'`
#/                    echo $uniqueName > $FILENAME_PREVIOUSRUN
#/                    ;;
#/                reuse)
#/                  line=$(head -n 1 $FILENAME_PREVIOUSRUN)
#/                  if [ -z $line  ]; then
#/                    echo "No previous runs yet !"
#/                    exit
#/                  else
#/                    uniqueName=$line
#/                  fi
#/                    ;;
#/                =*)
#/                    val=${OPTARG#*=}
#/                    opt=${OPTARG%=$val}
#/                    echo "parsing option: '--${opt}', value: '${val}'" >&2
#/                    ;;
#/                *)
#/                    if [ "$OPTERR" = 1 ] && [ "${optspec:0:1}" != ":" ]; then
#/                        echo "unknown option --${OPTARG}" >&2
#/                    fi
#/                    ;;
#/            esac;;
#/    \?)
#/      echo "Invalid option: -$OPTARG" >&2
#/      exit 1
#/      ;;
#/#    :)
#/#      echo "Option -$OPTARG requires an argument." >&2
#/#      exit 1
#/#      ;;
#/  esac
#/done
SUPPORTED_PLATFORMS="LINUX BGL BGP BGQ SMP"
#[[ $SUPPORTED_PLATFORMS =~ (^|[[:space:]])"$PLATFORM"($|[[:space:]]) ]] && echo 'right platform' || { echo "wrong-platform $PLATFORM"; exit; }
[[ $SUPPORTED_PLATFORMS =~ (^|[[:space:]])"$PLATFORM"($|[[:space:]]) ]] && : || { echo "wrong-platform $PLATFORM"; print_help; exit; }

SUPPORTED_COMPONENTS="all common mdl gsl nti"
[[ $SUPPORTED_COMPONENTS =~ (^|[[:space:]])"$WHAT_TO_BUILD"($|[[:space:]]) ]] && : || { echo "wrong component $WHAT_TO_BUILD"; print_help; exit; }

#}}}

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
if [ ! -f model.gsl ]; then
  echo "Please make sure model.gsl exist, e.g. run changemorph.sh"
  exit 1
fi
mv Topology.h .Topology_backup.h
echo "#define _X_ $_X_" > Topology.h
echo "#define _Y_ $_Y_" >> Topology.h
echo "#define _Z_ $_Z_" >> Topology.h
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
#mv $OutputFolderName $TMPDIR/
mv .Topology_backup.h Topology.h
#}}}

###########################
#{{{ 4. Final step
if [ "$secondArg" == "-comment" ]; then
  COMMENT=$thirdArg
fi
if [ ! -d "$OutputFolderName" ]; then
  mkdir $OutputFolderName
  RunSim
else
  if [ "$secondArg" != "-plot-only" ]; then
    echo "second arg is $secondArg"
    Yes_No_RunSim
  else
    RUNSIM_COMPLETED=1
  fi
fi

if [ "$secondArg" != "-no-plot" ]; then
  DoPlot
fi

DoFinish
rm ${temp_file}
#}}}
