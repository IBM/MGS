RECURSE=3
DISTROOT=./dist
mkdir $DISTROOT
cp $NTSROOT/setenv_nts $DISTROOT
cp $NTSROOT/make_nts $DISTROOT
mkdir $DISTROOT/mdl
mkdir $DISTROOT/mdl/bin
mkdir $DISTROOT/mdl/lib
mkdir $DISTROOT/mdl/obj
mkdir $DISTROOT/mdl/parser
mkdir $DISTROOT/mdl/scripts
mkdir $DISTROOT/mdl/include
mkdir $DISTROOT/mdl/scanner
mkdir $DISTROOT/mdl/src

cp $MDLROOT/sources.mk $DISTROOT/mdl
cp $MDLROOT/Makefile $DISTROOT/mdl
cp $MDLROOT/parser/mdl.y $DISTROOT/mdl/parser
cp $MDLROOT/scripts/*.* $DISTROOT/mdl/scripts
cp $MDLROOT/include/FlexLexer.h $DISTROOT/mdl/include
cp $MDLROOT/scanner/mdl.l $DISTROOT/mdl/scanner

mv $GSLROOT/utils/std/include/RNG.h $GSLROOT/utils/std/include/RNG.h.bak
sed 's/^#include/\/\/#include/g' $GSLROOT/utils/std/include/RNG.h.bak | sed 's/^typedef/\/\/typedef/g' | sed 's/\/\/#include "MRG32k3a.h"/#include "MRG32k3a.h"/g' | sed 's/\/\/typedef MRG32k3a/typedef MRG32k3a/g'  > $GSLROOT/utils/std/include/RNG.h 

sed -n '/\\/p' $DISTROOT/mdl/sources.mk | sed '/#/d' | sed 's/SRCS := //' | sed 's/ *\\$/.C $DISTROOT\/mdl\/src\//g' | sed 's/^/cp \$MDLROOT\/src\//' > tmpdist.sh
source ./tmpdist.sh

sed -n '/\\/p' $DISTROOT/mdl/sources.mk | sed '/#/d' | sed 's/^/sed -n '"'"'\/^\\#include \\\"\/p'"'"' $MDLROOT\/src\//' | sed 's/SRCS := //' | sed 's/ *\\$/.C | sed '"'"'s\/#include \\\"\/cp \\$MDLROOT\\\/include\\\/\/'"'"' | sed '"'"'s\/\\\"$\/ $DISTROOT\\\/mdl\\\/include\/'"'"'> tmpdist2.sh\
source \.\/tmpdist2.sh/g' > tmpdist.sh
source ./tmpdist.sh

sed -n '/\\/p' $DISTROOT/mdl/sources.mk | sed '/#/d' | sed 's/^/sed -n '"'"'\/^\\#include \\\"\/p'"'"' $MDLROOT\/include\//' | sed 's/SRCS := //' | sed 's/ *\\$/.h | sed '"'"'s\/#include \\\"\/cp \\$MDLROOT\\\/include\\\/\/'"'"' | sed '"'"'s\/\\\"$\/ $DISTROOT\\\/mdl\\\/include\/'"'"'> tmpdist2.sh\
source \.\/tmpdist2.sh/g' > tmpdist.sh
source ./tmpdist.sh

mkdir $DISTROOT/nti
mkdir $DISTROOT/gsl
mkdir $DISTROOT/gsl/lib
mkdir $DISTROOT/gsl/bin
mkdir $DISTROOT/gsl/scripts
mkdir $DISTROOT/gsl/so
mkdir $DISTROOT/gsl/extensions
mkdir $DISTROOT/gsl/extensions/edge
mkdir $DISTROOT/gsl/extensions/functor
mkdir $DISTROOT/gsl/extensions/interface
mkdir $DISTROOT/gsl/extensions/node
mkdir $DISTROOT/gsl/extensions/struct
mkdir $DISTROOT/gsl/extensions/subscriber
mkdir $DISTROOT/gsl/extensions/tool
mkdir $DISTROOT/gsl/extensions/trigger
mkdir $DISTROOT/gsl/extensions/variable
mkdir $DISTROOT/gsl/framework
mkdir $DISTROOT/gsl/framework/dataitems
mkdir $DISTROOT/gsl/framework/dataitems/include
mkdir $DISTROOT/gsl/framework/dataitems/obj
mkdir $DISTROOT/gsl/framework/dataitems/src
mkdir $DISTROOT/gsl/framework/dca
mkdir $DISTROOT/gsl/framework/dca/include
mkdir $DISTROOT/gsl/framework/dca/obj
mkdir $DISTROOT/gsl/framework/dca/src
mkdir $DISTROOT/gsl/framework/factories
mkdir $DISTROOT/gsl/framework/factories/include
mkdir $DISTROOT/gsl/framework/factories/obj
mkdir $DISTROOT/gsl/framework/factories/src
mkdir $DISTROOT/gsl/framework/functors
mkdir $DISTROOT/gsl/framework/functors/include
mkdir $DISTROOT/gsl/framework/functors/obj
mkdir $DISTROOT/gsl/framework/functors/src
mkdir $DISTROOT/gsl/framework/networks
mkdir $DISTROOT/gsl/framework/networks/include
mkdir $DISTROOT/gsl/framework/networks/obj
mkdir $DISTROOT/gsl/framework/networks/src
mkdir $DISTROOT/gsl/framework/parser
mkdir $DISTROOT/gsl/framework/parser/bison
mkdir $DISTROOT/gsl/framework/parser/flex
mkdir $DISTROOT/gsl/framework/parser/generated
mkdir $DISTROOT/gsl/framework/parser/include
mkdir $DISTROOT/gsl/framework/parser/obj
mkdir $DISTROOT/gsl/framework/parser/src
mkdir $DISTROOT/gsl/framework/simulation
mkdir $DISTROOT/gsl/framework/simulation/include
mkdir $DISTROOT/gsl/framework/simulation/obj
mkdir $DISTROOT/gsl/framework/simulation/src
mkdir $DISTROOT/models
mkdir $DISTROOT/graphs
mkdir $DISTROOT/gsl/utils
mkdir $DISTROOT/gsl/utils/img
mkdir $DISTROOT/gsl/utils/img/include
mkdir $DISTROOT/gsl/utils/img/obj
mkdir $DISTROOT/gsl/utils/img/src
mkdir $DISTROOT/gsl/utils/std
mkdir $DISTROOT/gsl/utils/std/include
mkdir $DISTROOT/gsl/utils/std/obj
mkdir $DISTROOT/gsl/utils/std/src
mkdir $DISTROOT/gsl/utils/streams
mkdir $DISTROOT/gsl/utils/streams/obj
mkdir $DISTROOT/gsl/utils/streams/include
mkdir $DISTROOT/gsl/utils/streams/src

cp models/compartmental.mdf $DISTROOT/models/
cp gsl/extensions/variable/variables.mdf $DISTROOT/gsl/extensions/variable/
cp gsl/extensions/struct/structs.mdf $DISTROOT/gsl/extensions/struct/
cp gsl/extensions/functor/functors.mdf $DISTROOT/gsl/extensions/functor/

cp gsl/framework/factories/include/Lens.h $DISTROOT/gsl/framework/factories/include/Lens.h

cp nti/Makefile $DISTROOT/nti
cp nti/MaxComputeOrder.h $DISTROOT/nti
sed -n '/\\/p' $DISTROOT/nti/Makefile | sed '/\$/d' | sed 's/^.*SOURCES = //' | sed 's/ *\\$/ $DISTROOT\/nti\//g' | sed 's/^/cp nti\//g' > tmpdist.sh
source ./tmpdist.sh
sed -n '/\\/p' nti/Makefile | sed '/\$/d' | sed 's/^.*SOURCES = //g' | sed 's/^/sed -n '"'"'\/^\\#include \\\"\/p'"'"' nti\//g' | sed 's/ *\\$/ | sed '"'"'s\/#include \\\"\/cp nti\\\/\/g'"'"' | sed '"'"'s\/\\\"\/ $DISTROOT\\\/nti\/g'"'"'> tmpdist2.sh\
source \.\/tmpdist2.sh/g' > tmpdist.sh
source ./tmpdist.sh
sed -n '/\\/p' nti/Makefile | sed '/\$/d' | sed 's/^.*SOURCES = //g' | sed 's/^/sed -n '"'"'\/^\\#include \\\"\/p'"'"' nti\//g' | sed 's/\.cxx/\.h/g' | sed 's/ *\\$/ | sed '"'"'s\/#include \\\"\/cp nti\\\/\/g'"'"' | sed '"'"'s\/\\\"\/ $DISTROOT\\\/nti\/g'"'"'> tmpdist2.sh\
source \.\/tmpdist2.sh/g' > tmpdist.sh
source ./tmpdist.sh
cp nti/CountableModel.h $DISTROOT/nti
cp nti/TableEntry.h $DISTROOT/nti
rm -f $DISTROOT/nti?

cp gsl/configure.py $DISTROOT/gsl
cp gsl/clean.sh $DISTROOT/gsl
cp gsl/scripts/depend.sh $DISTROOT/gsl/scripts
cp gsl/framework/parser/bison/speclang.y $DISTROOT/gsl/framework/parser/bison
cp gsl/framework/parser/flex/speclang.l $DISTROOT/gsl/framework/parser/flex
cp gsl/framework/dca/src/fakesocket.c $DISTROOT/gsl/framework/dca/src
cp gsl/framework/networks/include/DemarshallerInstance.h $DISTROOT/gsl/framework/networks/include
cp gsl/framework/networks/include/VariableProxyBase.h $DISTROOT/gsl/framework/networks/include
cp gsl/framework/networks/include/Marshall.h $DISTROOT/gsl/framework/networks/include
cp gsl/utils/std/include/DeepPointerArray.h $DISTROOT/gsl/utils/std/include
cp gsl/framework/parser/include/FlexFixer.h $DISTROOT/gsl/framework/parser/include
cp gsl/framework/parser/include/FlexLexer.h $DISTROOT/gsl/framework/parser/include
cp gsl/framework/parser/flex/lex.yy.C.linux.i386 $DISTROOT/gsl/framework/parser/flex
cp mdl/include/FlexFixer.h $DISTROOT/mdl/include

mkdir $DISTROOT/graphs/Compartmental
mkdir $DISTROOT/graphs/Compartmental/neurons
cp graphs/Compartmental/Tissue1.gsl $DISTROOT/graphs/Compartmental/Tissue1.gsl
cp graphs/Compartmental/Tissue2.gsl $DISTROOT/graphs/Compartmental/Tissue2.gsl
cp graphs/Compartmental/Ca-Tissue.gsl $DISTROOT/graphs/Compartmental/Ca-Tissue.gsl
cp graphs/Compartmental/CptParams1.par $DISTROOT/graphs/Compartmental
cp graphs/Compartmental/CptParams2.par $DISTROOT/graphs/Compartmental
cp graphs/Compartmental/CptParamsCa.par $DISTROOT/graphs/Compartmental
cp graphs/Compartmental/ChanParams1.par $DISTROOT/graphs/Compartmental
cp graphs/Compartmental/ChanParams2.par $DISTROOT/graphs/Compartmental
cp graphs/Compartmental/ChanParamsCa.par $DISTROOT/graphs/Compartmental
cp graphs/Compartmental/SynParams1.par $DISTROOT/graphs/Compartmental
cp graphs/Compartmental/SynParams2.par $DISTROOT/graphs/Compartmental
cp graphs/Compartmental/SynParamsCa.par $DISTROOT/graphs/Compartmental
cp graphs/Compartmental/DevParams.par $DISTROOT/graphs/Compartmental
cp graphs/Compartmental/DetParams.par $DISTROOT/graphs/Compartmental
cp graphs/Compartmental/minicolumn.txt $DISTROOT/graphs/Compartmental/
cp graphs/Compartmental/Topology.h $DISTROOT/graphs/Compartmental/

#cp graphs/Compartmental/neurons/minicolumn/C050896A-P3.CNG.swc $DISTROOT/graphs/Compartmental/neurons/minicolumn
#cp graphs/Compartmental/neurons/minicolumn/C261296A-P1.CNG.swc $DISTROOT/graphs/Compartmental/neurons/minicolumn
#cp graphs/Compartmental/neurons/minicolumn/C261296A-P2.CNG.swc $DISTROOT/graphs/Compartmental/neurons/minicolumn
#cp graphs/Compartmental/neurons/minicolumn/C261296A-P3.CNG.swc $DISTROOT/graphs/Compartmental/neurons/minicolumn
#cp graphs/Compartmental/neurons/minicolumn/C040896A-P3.CNG.swc $DISTROOT/graphs/Compartmental/neurons/minicolumn
#cp graphs/Compartmental/neurons/minicolumn/C120398A-P3.CNG.swc $DISTROOT/graphs/Compartmental/neurons/minicolumn
#cp graphs/Compartmental/neurons/minicolumn/C010600A2.CNG.swc $DISTROOT/graphs/Compartmental/neurons/minicolumn
#cp graphs/Compartmental/neurons/minicolumn/C050800E2.CNG.swc $DISTROOT/graphs/Compartmental/neurons/minicolumn
#cp graphs/Compartmental/neurons/minicolumn/C200897C-I1.CNG.swc $DISTROOT/graphs/Compartmental/neurons/minicolumn
#cp graphs/Compartmental/neurons/minicolumn/C120398A-P2.CNG.swc $DISTROOT/graphs/Compartmental/neurons/minicolumn
#cp graphs/Compartmental/neurons/minicolumn/C010600B1.CNG.swc $DISTROOT/graphs/Compartmental/neurons/minicolumn
#cp graphs/Compartmental/neurons/minicolumn/C120398A-P1.CNG.swc $DISTROOT/graphs/Compartmental/neurons/minicolumn
#cp graphs/Compartmental/neurons/minicolumn/C190898A-P2.CNG.swc $DISTROOT/graphs/Compartmental/neurons/minicolumn
#cp graphs/Compartmental/neurons/minicolumn/C190898A-P3.CNG.swc $DISTROOT/graphs/Compartmental/neurons/minicolumn
#cp graphs/Compartmental/neurons/minicolumn/C250500A-I4.CNG.swc $DISTROOT/graphs/Compartmental/neurons/minicolumn
#cp graphs/Compartmental/neurons/minicolumn/C180298A-P2.CNG.swc $DISTROOT/graphs/Compartmental/neurons/minicolumn
#cp graphs/Compartmental/neurons/minicolumn/C040600A2.CNG.swc $DISTROOT/graphs/Compartmental/neurons/minicolumn
#cp graphs/Compartmental/neurons/minicolumn/C240797B-P3.CNG.swc $DISTROOT/graphs/Compartmental/neurons/minicolumn
#cp graphs/Compartmental/neurons/minicolumn/C050600B1.CNG.swc $DISTROOT/graphs/Compartmental/neurons/minicolumn 
#cp graphs/Compartmental/neurons/minicolumn/C280199C-P1.CNG.swc $DISTROOT/graphs/Compartmental/neurons/minicolumn
              
mkdir $DISTROOT/models/std
mkdir $DISTROOT/models/LaboratoryTools
mkdir $DISTROOT/models/HodgkinHuxley
mkdir $DISTROOT/models/HodgkinHuxleyVoltage
mkdir $DISTROOT/models/HodgkinHuxleyVoltageJunction
mkdir $DISTROOT/models/CaConcentration
mkdir $DISTROOT/models/CaConcentrationJunction
mkdir $DISTROOT/models/NaChannel
mkdir $DISTROOT/models/NaChannel_AIS
mkdir $DISTROOT/models/KDRChannel
mkdir $DISTROOT/models/KDRChannel_AIS
mkdir $DISTROOT/models/KDRChannel_IO
mkdir $DISTROOT/models/CahChannel
mkdir $DISTROOT/models/CalChannel
mkdir $DISTROOT/models/KCaChannel
mkdir $DISTROOT/models/HCNChannel
mkdir $DISTROOT/models/Connexon
mkdir $DISTROOT/models/CaConnexon
mkdir $DISTROOT/models/PreSynapticPoint
mkdir $DISTROOT/models/GABAAReceptor
mkdir $DISTROOT/models/AMPAReceptor
mkdir $DISTROOT/models/NMDAReceptor
mkdir $DISTROOT/models/BranchSolver

cp models/std/std.mdl $DISTROOT/models/std
cp models/LaboratoryTools/LaboratoryTools.mdl /$DISTROOT/models/LaboratoryTools
cp models/HodgkinHuxley/HodgkinHuxley.mdl $DISTROOT/models/HodgkinHuxley
cp models/HodgkinHuxleyVoltage/HodgkinHuxleyVoltage.mdl $DISTROOT/models/HodgkinHuxleyVoltage
cp models/HodgkinHuxleyVoltageJunction/HodgkinHuxleyVoltageJunction.mdl $DISTROOT/models/HodgkinHuxleyVoltageJunction
cp models/CaConcentration/CaConcentration.mdl $DISTROOT/models/CaConcentration
cp models/CaConcentrationJunction/CaConcentrationJunction.mdl $DISTROOT/models/CaConcentrationJunction
cp models/NaChannel/NaChannel.mdl $DISTROOT/models/NaChannel
cp models/NaChannel_AIS/NaChannel_AIS.mdl $DISTROOT/models/NaChannel_AIS
cp models/KDRChannel/KDRChannel.mdl $DISTROOT/models/KDRChannel
cp models/KDRChannel_AIS/KDRChannel_AIS.mdl $DISTROOT/models/KDRChannel_AIS
cp models/KDRChannel_IO/KDRChannel_IO.mdl $DISTROOT/models/KDRChannel_IO
cp models/CahChannel/CahChannel.mdl $DISTROOT/models/CahChannel
cp models/CalChannel/CalChannel.mdl $DISTROOT/models/CalChannel
cp models/KCaChannel/KCaChannel.mdl $DISTROOT/models/KCaChannel
cp models/HCNChannel/HCNChannel.mdl $DISTROOT/models/HCNChannel
cp models/Connexon/Connexon.mdl $DISTROOT/models/Connexon
cp models/CaConnexon/CaConnexon.mdl $DISTROOT/models/CaConnexon
cp models/PreSynapticPoint/PreSynapticPoint.mdl $DISTROOT/models/PreSynapticPoint
cp models/GABAAReceptor/GABAAReceptor.mdl $DISTROOT/models/GABAAReceptor
cp models/AMPAReceptor/AMPAReceptor.mdl $DISTROOT/models/AMPAReceptor
cp models/NMDAReceptor/NMDAReceptor.mdl $DISTROOT/models/NMDAReceptor
cp models/BranchSolver/BranchSolver.mdl $DISTROOT/models/BranchSolver
cp models/define $DISTROOT/models/

cp gsl/extensions/variable/define $DISTROOT/gsl/extensions/variable/
cp gsl/extensions/struct/define $DISTROOT/gsl/extensions/struct/
cp gsl/extensions/functor/define $DISTROOT/gsl/extensions/functor/

for NODETYPE in HodgkinHuxleyVoltage VoltageEndPoint HodgkinHuxleyVoltageJunction VoltageJunctionPoint CaConcentration CaConcentrationEndPoint CaConcentrationJunction CaConcentrationJunctionPoint NaChannel NaChannel_AIS KDRChannel KDRChannel_AIS KDRChannel_IO CahChannel CalChannel KCaChannel HCNChannel Connexon CaConnexon PreSynapticPoint GABAAReceptor AMPAReceptor NMDAReceptor ForwardSolvePoint1 ForwardSolvePoint2 ForwardSolvePoint3 ForwardSolvePoint4 ForwardSolvePoint5 ForwardSolvePoint6 ForwardSolvePoint7 BackwardSolvePoint0 BackwardSolvePoint1 BackwardSolvePoint2 BackwardSolvePoint3 BackwardSolvePoint4 BackwardSolvePoint5 BackwardSolvePoint6
do
  mkdir $DISTROOT/gsl/extensions/node/$NODETYPE
  mkdir $DISTROOT/gsl/extensions/node/$NODETYPE/src
  mkdir $DISTROOT/gsl/extensions/node/$NODETYPE/include
  cp gsl/extensions/node/$NODETYPE/src/$NODETYPE.C $DISTROOT/gsl/extensions/node/$NODETYPE/src
  cp gsl/extensions/node/$NODETYPE/src/$NODETYPE\CompCategory.C $DISTROOT/gsl/extensions/node/$NODETYPE/src
  cp gsl/extensions/node/$NODETYPE/include/$NODETYPE.h $DISTROOT/gsl/extensions/node/$NODETYPE/include
  cp gsl/extensions/node/$NODETYPE/include/$NODETYPE\CompCategory.h $DISTROOT/gsl/extensions/node/$NODETYPE/include
done

for PTH in gsl/utils/std gsl/utils/img gsl/utils/streams gsl/framework/dataitems gsl/framework/dca gsl/framework/factories gsl/framework/functors gsl/framework/networks gsl/framework/parser gsl/framework/simulation
do
  cp $PTH/module.mk $DISTROOT/$PTH
  for PTH2 in gsl/utils/std gsl/utils/img gsl/utils/streams gsl/framework/dataitems gsl/framework/dca gsl/framework/factories gsl/framework/functors gsl/framework/networks gsl/framework/parser gsl/framework/simulation
    do
    sed -n '/\\/p' $DISTROOT/$PTH/module.mk | sed '/#/d' | sed 's/SOURCES := //' | sed 's/ *\\$/ $DISTROOT\/\$PTH\/src\//g' | sed 's/^/cp \$PTH\/src\//g' > tmpdist.sh
    source ./tmpdist.sh
    sed -n '/\\/p' $PTH/module.mk | sed '/#/d' | sed 's/^/sed -n '"'"'\/^\\#include \\\"\/p'"'"' \$PTH\/src\//g' | sed 's/SOURCES := //g' | sed 's/ *\\$/ | sed '"'"'s\/#include \\\"\/cp \\\$PTH2\\\/include\\\/\/g'"'"' | sed '"'"'s\/\\\"\/ $DISTROOT\\\/\$PTH2\\\/include\/g'"'"'> tmpdist2.sh\
source \.\/tmpdist2.sh/g' > tmpdist.sh
    source ./tmpdist.sh
    sed -n '/\\/p' $PTH/module.mk | sed '/#/d' | sed 's/^/sed -n '"'"'\/^\\#include \\\"\/p'"'"' \$PTH\/include\//g' | sed 's/SOURCES := //g' | sed 's/\.C/\.h/g' | sed 's/ *\\$/ | sed '"'"'s\/#include \\\"\/cp \\\$PTH2\\\/include\\\/\/g'"'"' | sed '"'"'s\/\\\"\/ $DISTROOT\\\/\$PTH2\\\/include\/g'"'"'> tmpdist2.sh\
source \.\/tmpdist2.sh/g' > tmpdist.sh
    source ./tmpdist.sh
  done
done

echo "gsl/extensions/variable/CurrentDisplay/src/CurrentDisplay.C
gsl/extensions/variable/ConductanceDisplay/src/ConductanceDisplay.C
gsl/extensions/variable/VoltageDisplay/src/VoltageDisplay.C
gsl/extensions/variable/CalciumVisualization/src/CalciumVisualization.C
gsl/extensions/variable/VoltageVisualization/src/VoltageVisualization.C
gsl/extensions/variable/VoltageClamp/src/VoltageClamp.C
gsl/extensions/variable/CaCurrentDisplay/src/CaCurrentDisplay.C
gsl/extensions/variable/CalciumDisplay/src/CalciumDisplay.C
gsl/extensions/variable/CurrentPulseGenerator/src/CurrentPulseGenerator.C
gsl/extensions/variable/SimulationSetter/src/SimulationSetter.C
gsl/extensions/variable/PointCurrentSource/src/PointCurrentSource.C
gsl/extensions/variable/PointCalciumSource/src/PointCalciumSource.C
gsl/extensions/variable/ReversalPotentialDisplay/src/ReversalPotentialDisplay.C" > gsl/extensions/variable/module.C.ext;
echo "" > gsl/extensions/struct/module.C.ext;
echo "" > gsl/extensions/functor/module.C.ext;

echo "gsl/extensions/variable/CurrentDisplay/include/CurrentDisplay.h
gsl/extensions/variable/ConductanceDisplay/include/ConductanceDisplay.h
gsl/extensions/variable/VoltageDisplay/include/VoltageDisplay.h
gsl/extensions/variable/VoltageVisualization/include/VoltageVisualization.h
gsl/extensions/variable/CalciumVisualization/include/CalciumVisualization.h
gsl/extensions/variable/VoltageClamp/include/VoltageClamp.h
gsl/extensions/variable/CalciumDisplay/include/CalciumDisplay.h
gsl/extensions/variable/CaCurrentDisplay/include/CaCurrentDisplay.h
gsl/extensions/variable/CurrentPulseGenerator/include/CurrentPulseGenerator.h
gsl/extensions/variable/SimulationSetter/include/SimulationSetter.h
gsl/extensions/variable/PointCurrentSource/include/PointCurrentSource.h
gsl/extensions/variable/PointCalciumSource/include/PointCalciumSource.h
gsl/extensions/variable/ReversalPotentialDisplay/include/ReversalPotentialDisplay.h" > gsl/extensions/variable/module.h.ext;
echo "" > gsl/extensions/struct/module.h.ext;
echo "" > gsl/extensions/functor/module.h.ext;

for EXT_TYP in variable struct functor
  do
  cp gsl/extensions/$EXT_TYP/$EXT_TYP\s.mdf tmp1.ext
  sed '/\\/p' tmp1.ext | sed 's/#include "/cp gsl\/extensions\/'$EXT_TYP'\//g' | sed 's/"/ '\$DISTROOT'\/gsl\/extensions\/'$EXT_TYP'\//g' > tmp2.ext
  source tmp2.ext;
  sed '/\\/p' tmp1.ext | sed 's/#include "/find gsl\/extensions\/'$EXT_TYP'\//g' | sed 's/.mdl"/ -name *.C > tmp3.ext ; cat gsl\/extensions\/'$EXT_TYP'\/module.C.ext tmp3.ext > tmp4.ext; cp tmp4.ext gsl\/extensions\/'$EXT_TYP'\/module.C.ext;/g' > tmp2.ext
  source tmp2.ext;
  sed '/CG_/d' gsl/extensions/$EXT_TYP/module.C.ext | sed 's/'$EXT_TYP'//g' | sed 's/src//g' | sed 's/\/\// /g' | awk '{print "mkdir '$DISTROOT'/gsl/extensions/'$EXT_TYP'/" $2}' > tmpdist.sh
  source ./tmpdist.sh
  sed '/CG_/d' gsl/extensions/$EXT_TYP/module.C.ext | sed 's/'$EXT_TYP'//g' | sed 's/src//g' | sed 's/\/\// /g' | awk '{print "mkdir '$DISTROOT'/gsl/extensions/'$EXT_TYP'/" $2 "/src"}' > tmpdist.sh
  source ./tmpdist.sh
  sed '/CG_/d' gsl/extensions/$EXT_TYP/module.C.ext | awk '{print "cp " $1 " '$DISTROOT'/" $1}' > tmpdist.sh
  source ./tmpdist.sh
  sed '/CG_/d' gsl/extensions/$EXT_TYP/module.C.ext | sed 's/'$EXT_TYP'//g' | sed 's/src//g' | sed 's/\/\// /g' | awk '{print "mkdir '$DISTROOT'/gsl/extensions/'$EXT_TYP'/" $2 "/include"}' > tmpdist.sh
  source ./tmpdist.sh

  sed '/\\/p' tmp1.ext | sed 's/#include "/find gsl\/extensions\/'$EXT_TYP'\//g' | sed 's/.mdl"/ -name *.h > tmp3.ext ; cat gsl\/extensions\/'$EXT_TYP'\/module.h.ext tmp3.ext > tmp4.ext; cp tmp4.ext gsl\/extensions\/'$EXT_TYP'\/module.h.ext;/g' > tmp2.ext
  source tmp2.ext;
  rm -f tmp*.ext
  sed '/CG_/d' gsl/extensions/$EXT_TYP/module.h.ext | awk '{print "cp " $1 " '$DISTROOT'/" $1}' > tmpdist.sh
  source ./tmpdist.sh
  for PTH2 in gsl/utils/std gsl/utils/img gsl/utils/streams gsl/framework/dataitems gsl/framework/dca gsl/framework/factories gsl/framework/functors gsl/framework/networks gsl/framework/parser gsl/framework/simulation
    do
    ls -d1 $DISTROOT/gsl\/extensions/$EXT_TYP/*/src/*.C > recurse.out
    awk '{print "sed -n '"'"'/^#include \\\"/p'"'"' " $1 " | sed '"'"'s/#include \\\"/cp gsl\\/extensions\\/$EXT_TYP2\\/include\\//g'"'"' | sed '"'"'s/\\\"/ $DISTROOT\\/gsl\\/extensions\\/$EXT_TYP2\\/include\\//g'"'"' > tmpdist2.sh\nsource ./tmpdist2.sh"}' recurse.out > tmpdist.sh
    source ./tmpdist.sh
    ls -d1 $DISTROOT/gsl/extensions/$EXT_TYP/*/include/*.h > recurse.out
    awk '{print "sed -n '"'"'/^#include \\\"/p'"'"' " $1 " | sed '"'"'s/#include \\\"/cp gsl\\/extensions\\/$EXT_TYP2\\/include\\//g'"'"' | sed '"'"'s/\\\"/ $DISTROOT\\/gsl\\/extensions\\/$EXT_TYP2\\/include\\//g'"'"' > tmpdist2.sh\nsource ./tmpdist2.sh"}' recurse.out > tmpdist.sh
    source ./tmpdist.sh
  done
  rm -f module.C.ext
  rm -f module.h.ext
done

for PTH in gsl/utils/std gsl/utils/img gsl/utils/streams gsl/framework/dataitems gsl/framework/dca gsl/framework/factories gsl/framework/functors gsl/framework/networks gsl/framework/parser gsl/framework/simulation
do
  for PTH2 in gsl/utils/std gsl/utils/img gsl/utils/streams gsl/framework/dataitems gsl/framework/dca gsl/framework/factories gsl/framework/functors gsl/framework/networks gsl/framework/parser gsl/framework/simulation
    do
    echo $PTH2
    for ((i=0; i<RECURSE; i++))
      do
      ls -a1 $DISTROOT/$PTH/include/*.h > recurse.out
      awk '{print "sed -n '"'"'/^#include \\\"/p'"'"' " $1 " | sed '"'"'s/#include \\\"/cp $PTH2\\/include\\//g'"'"' | sed '"'"'s/\\\"/ $DISTROOT\\/$PTH2\\/include\\//g'"'"' > tmpdist2.sh\nsource ./tmpdist2.sh"}' recurse.out > tmpdist.sh
      source ./tmpdist.sh
    done
  done
done

rm -f tmpdist.sh tmpdist2.sh

mv $GSLROOT/utils/std/include/RNG.h.bak $GSLROOT/utils/std/include/RNG.h

cp $NTSROOT/nti/FrontLimitedSegmentSpace.cxx $DISTROOT/nti
cp $NTSROOT/nti/InferiorOliveGlomeruliDetector.h $DISTROOT/nti
cp $NTSROOT/nti/GlomeruliDetector.h $DISTROOT/nti
cp $NTSROOT/nti/touchDetect.cxx $DISTROOT/nti
cp $NTSROOT/nti/composeSwc.cxx $DISTROOT/nti
cp $NTSROOT/models/LaboratoryTools/LaboratoryTools.mdl $DISTROOT/models/LaboratoryTools