#!/bin/bash
#author  = Hoang Trong Minh Tuan (@IBM - 2016)
#version = 1.0
#///{{{
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
   echo "----> $OutputFolderName" >> SIM_LOG
   echo "----> RESULT: " >> SIM_LOG
   echo "---------------------- " >> SIM_LOG
   cp SIM_LOG $OutputFolderName/ -L -r
   echo "Output Folder: " $OutputFolderName
   mpiexec -n 2  ../../gsl/bin/gslparser $temp_file -t 4
   echo "Output Folder: " $OutputFolderName
   ## NOTE: comment out if we don't want to plot
   ./doPlot.sh  ${uniqueName:1} ${runCaseNumber}
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
## 1. CREATE FILE/FOLDER
TMPDIR=`pwd`/.tmp
if [ ! -d $TMPDIR ]; then
  mkdir $TMPDIR
fi
FILENAME_PREVIOUSRUN=$TMPDIR/previousRun
if [ ! -f $FILENAME_PREVIOUSRUN ]; then
  touch $FILENAME_PREVIOUSRUN
fi
temp_file=`mktemp --tmpdir=$TMPDIR`

## 2. CHECK ARGS
##{{{
if [ "$#" == "0" ]; then
  echo "$0 <extension> "
  echo "    <extension> somename to make the output folder unique"
  echo "     You can pass something like -unique 
    then the script generate a unique name using `date +'%Y-%m-%d-%s'` which evoke the date"
  echo "Example: "
  echo " $0 -unique"
  echo " $0 abc"
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
## 3. CHECK MACROS
cpp -dU -P model.gsl -DEXTENSION=$uniqueName > ${temp_file} 2> /dev/null
###cpp -dU -P model.gsl -DEXTENSION=$uniqueName > out.txt 2> /dev/null
##awk -F '/^#define[[:space:]]+morph/{ printf "%s | %s \n", $2, $3 }' < ${temp_file}
#morph=`awk  '/^#define morph/{printf "%s\n", $3}' < ${temp_file}`
morph=`awk  '/^#define morph/{print $3}' < ${temp_file}`
#dataFolder=`awk  '/^#define dataFolder/{printf "%s\n", $3}' < ${temp_file}`
dataFolder=`awk  '/^#define dataFolder/{print $3}' < ${temp_file}`
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

###########################
if [ ! -d $OutputFolderName ]; then
  mkdir $OutputFolderName
  RunSim
else
  Yes_No_RunSim
fi

DoFinish
rm ${temp_file}
