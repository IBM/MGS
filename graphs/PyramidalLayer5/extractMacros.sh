#!/bin/bash
#///{{{
Yes_No_RunSim()
{
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
}
RunSim()
{
   echo "Output Folder: " $OutputFolderName
   cp params $OutputFolderName/ -L -r
   cp *.gsl $OutputFolderName/ -L -r
   mpiexec -n 2  ../../gsl/bin/gslparser $temp_file -t 4
   echo "Output Folder: " $OutputFolderName
}
#///}}}

#########################
## CHECK ARGS
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
  uniqueName=`date +'%Y-%m-%d-%s'`
else
  uniqueName=$1
fi
#}}}
#########################
##
temp_file=$(mktemp)
cpp -dU -P model.gsl -DEXTENSION=$1 > ${temp_file} 2> /dev/null
##awk -F '/^#define[[:space:]]+morph/{ printf "%s | %s \n", $2, $3 }' < ${temp_file}
#morph=`awk  '/^#define morph/{printf "%s\n", $3}' < ${temp_file}`
morph=`awk  '/^#define morph/{print $3}' < ${temp_file}`
#dataFolder=`awk  '/^#define dataFolder/{printf "%s\n", $3}' < ${temp_file}`
dataFolder=`awk  '/^#define dataFolder/{print $3}' < ${temp_file}`
suffix=`awk  '/^#define OutputFolderName/{ printf "%s \n", $5 }' < ${temp_file}`
OutputFolderName=`echo $dataFolder | sed -e 's/^"//' -e 's/"$//'`
OutputFolderName+=`echo $morph | sed -e 's/^"//' -e 's/"$//'`
OutputFolderName+=`echo $suffix | sed -e 's/^"//' -e 's/"$//'`
OutputFolderName+=$uniqueName
##echo "$morph" | sed -e 's/^"//' -e 's/"$//'
#echo "morph="$morph
#echo $dataFolder
#echo $suffix

###########################
if [ ! -d $OutputFolderName ]; then
  mkdir $OutputFolderName
  RunSim
else
  Yes_No_RunSim
fi
rm ${temp_file}
