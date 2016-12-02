#include "../../../nti/include/MaxComputeOrder.h"
#include "../../std/std.gsl"
#include "Topology.h"

#define _CCAT(x,y) x ## y 
#define CCAT(x,y) _CCAT(x,y)
#define _STR(x) #x 
#define STR(x) _STR(x)
#define DAT0 "soma_voltage.dat"
#define DAT1 "axon_voltage.dat"
#define DAT2 "dend_voltage.dat"
//#define DAT3 "rec3.dat"
//#define DAT4 "rec4.dat"
//#define DAT5 "rec5.dat"
//#define DAT6 "rec6.dat"
//#define DAT7 "rec7.dat"
//#define DAT8 "rec8.dat"
//#define DAT9 "rec9.dat"
//#define DAT10 "rec10.dat"
//#define DAT11 "rec11.dat"
//#define DAT12 "rec12.dat"
//#define DAT13 "rec13.dat"
//#define DAT14 "rec14.dat"
//
//#define DAT20 "rec20.dat"
//#define DAT21 "rec21.dat"
//#define DAT22 "rec22.dat"
//#define DAT23 "rec23.dat"
//#define DAT24 "rec24.dat"
//#define DAT25 "rec25.dat"
//#define DAT26 "rec26.dat"
//#define DAT27 "rec27.dat"
//#define DAT28 "rec28.dat"
//#define DAT29 "rec29.dat"
//#define DAT30 "rec30.dat"
//#define DAT31 "rec31.dat"
//#define DAT32 "rec32.dat"
//#define DAT33 "rec33.dat"
//#define DAT34 "rec34.dat"

#define CA_BETA 0.05
#define CA_CLEARANCE 0.025
#define CA_DCA 0.400

#define V_CM 0.01
#define V_RA 0.001
#define V_NACONC 70.0
#define V_KCONC 325.0
#define V_GLEAK 0.00015
#define V_E_LEAK -10.0

#define PRMASK0 CATEGORY="BRANCH", TYPE="Voltage", BRANCHTYPE=3, MTYPE=0, NEURON_INDEX=0
#define PRMASK1 CATEGORY="BRANCH", TYPE="Voltage", BRANCHTYPE=3, MTYPE=0, NEURON_INDEX=1

Connector Zipper();
Zipper zipper();

Layout TissueLayoutFunctor();
		TissueLayoutFunctor tissueLayoutFunctor();		

NodeInitializer TissueNodeInitFunctor();
TissueNodeInitFunctor tissueNodeInitFunctor();

Connector TissueConnectorFunctor();
TissueConnectorFunctor tissueConnectorFunctor();

Functor TissueProbeFunctor();
TissueProbeFunctor tissueProbeFunctor();

Functor TissueMGSifyFunctor();
TissueMGSifyFunctor tissueMGSifyFunctor();

Functor TissueFunctor(string commandLine, string commandLineModification,	
		      string channelParamaterFile, string synapseParameteurFile,
		      Functor, Functor, Functor, Functor, Functor);

TissueFunctor tissueFunctor("-i IO_5_neurons_sorted.txt -j 1 -a 1.0 -u bt -o single_neuron_developed.txt -r 1 -p DevParamsIO.par -x " 
	      		    STR(_X_) " -y " STR(_Y_) " -z " STR(_Z_) " -e 0.001 -t 0.01 -m 0", 
			    "-p DetParamsIO.par -m 1 -n cost-volume",
			    "CptParamsIO.par", "ChanParamsIO.par", "SynParamsIO.par",
			    tissueLayoutFunctor, tissueNodeInitFunctor, 
                            tissueConnectorFunctor, tissueProbeFunctor);

GranuleMapper GridGranuleMapper(string description, list<int> dimensions, list<int> densityVector);
GridGranuleMapper tissueGM("Tissue Grid's GridGranuleMapper", { _X_ , _Y_ , _Z_ }, {1});
	
InitPhases = { initialize1, initialize2, initialize3 };

RuntimePhases = { solveChannels, predictJunction,
#if MAX_COMPUTE_ORDER>6
		  forwardSolve7,
#endif
#if MAX_COMPUTE_ORDER>5  
		  forwardSolve6,
#endif
#if MAX_COMPUTE_ORDER>4
		  forwardSolve5,	
#endif
#if MAX_COMPUTE_ORDER>3
		  forwardSolve4,
#endif
#if MAX_COMPUTE_ORDER>2
		  forwardSolve3,
#endif
#if MAX_COMPUTE_ORDER>1
		  forwardSolve2,
#endif
#if MAX_COMPUTE_ORDER>0	
		  forwardSolve1,
#endif
		  solve, 
#if MAX_COMPUTE_ORDER>0	
	          backwardSolve1,
#endif
#if MAX_COMPUTE_ORDER>1
	          backwardSolve2,
#endif	
#if MAX_COMPUTE_ORDER>2
	          backwardSolve3,
#endif
#if MAX_COMPUTE_ORDER>3
	          backwardSolve4,
#endif
#if MAX_COMPUTE_ORDER>4
	          backwardSolve5,
#endif
#if MAX_COMPUTE_ORDER>5
	          backwardSolve6,
#endif
#if MAX_COMPUTE_ORDER>6
	          backwardSolve7,
#endif
		  correctJunction, finish };
FinalPhases = { finalize };

NodeType HodgkinHuxleyVoltage (<
			Ra=V_RA,		// Gohm*um
			Na=V_NACONC, 		// mM
			K=V_KCONC,		// mM	
			E_leak=V_E_LEAK		// mV
			     >) { initializeCompartmentData->initialize2 };

NodeType CaConcentration (<
                        DCa=CA_DCA,                 // um^2/ms unbuffered
                        beta=CA_BETA                // dimensionless, Wagner and Keizer buffering constant
		     >) { initializeCompartmentData->initialize2, deriveParameters->initialize2 };

NodeType VoltageEndPoint { produceInitialState->initialize1,
#if MAX_COMPUTE_ORDER>0
		    produceSolvedVoltage-> CCAT(backwardSolve,MAX_COMPUTE_ORDER),
#else
		    produceSolvedVoltage->solve,
#endif
		    produceFinishedVoltage->finish
		  };

NodeType CaConcentrationEndPoint { produceInitialState->initialize1,
#if MAX_COMPUTE_ORDER>0
		    produceSolvedCaConcentration-> CCAT(backwardSolve,MAX_COMPUTE_ORDER),
#else
		    produceSolvedCaConcentration->solve,
#endif
		    produceFinishedCaConcentration->finish
		  };

#if MAX_COMPUTE_ORDER>0
NodeType BackwardSolvePoint0 { produceInitialState->initialize1,
		    produceBackwardSolution->solve };
NodeType ForwardSolvePoint1 { produceInitialState->initialize1,
		    produceInitialCoefficients->initialize2,
		    produceForwardSolution->forwardSolve1 };
#endif

#if MAX_COMPUTE_ORDER>1
NodeType BackwardSolvePoint1 { produceInitialState->initialize1,
		    produceBackwardSolution->backwardSolve1 };
NodeType ForwardSolvePoint2 { produceInitialState->initialize1,
		    produceInitialCoefficients->initialize2,
		    produceForwardSolution->forwardSolve2 };
#endif

#if MAX_COMPUTE_ORDER>2
NodeType BackwardSolvePoint2 { produceInitialState->initialize1,
		    produceBackwardSolution->backwardSolve2 };
NodeType ForwardSolvePoint3 { produceInitialState->initialize1,
		    produceInitialCoefficients->initialize2,
		    produceForwardSolution->forwardSolve3 };
#endif

#if MAX_COMPUTE_ORDER>3

NodeType BackwardSolvePoint3 { produceInitialState->initialize1,
		    produceBackwardSolution->backwardSolve3 };
NodeType ForwardSolvePoint4 { produceInitialState->initialize1,
		    produceInitialCoefficients->initialize2,
		    produceForwardSolution->forwardSolve4 };
#endif

#if MAX_COMPUTE_ORDER>4
NodeType BackwardSolvePoint4 { produceInitialState->initialize1,
		    produceBackwardSolution->backwardSolve4 };
NodeType ForwardSolvePoint5 { produceInitialState->initialize1,
		    produceInitialCoefficients->initialize2,
		    produceForwardSolution->forwardSolve5 };
#endif

#if MAX_COMPUTE_ORDER>5
NodeType BackwardSolvePoint5 { produceInitialState->initialize1,
		    produceBackwardSolution->backwardSolve5 };
NodeType ForwardSolvePoint6 { produceInitialState->initialize1,
			    produceInitialCoefficients->initialize2,
		    produceForwardSolution->forwardSolve6 };
#endif

#if MAX_COMPUTE_ORDER>6
NodeType BackwardSolvePoint6 { produceInitialState->initialize1,
		    produceBackwardSolution->backwardSolve6 };
NodeType ForwardSolvePoint7 { produceInitialState->initialize1,
		    produceInitialCoefficients->initialize2,
		    produceForwardSolution->forwardSolve7 };
#endif

NodeType HodgkinHuxleyVoltageJunction (<
			Ra=V_RA,		// Gohm*um
			Na=V_NACONC, 		// mM
			K=V_KCONC,		// mM	
			E_leak=V_E_LEAK		// mV
 		    >) { initializeJunction->initialize2 };

NodeType CaConcentrationJunction (<
                        DCa=CA_DCA,                 // um^2/ms unbuffered
                        beta=CA_BETA                // dimensionless, Wagner and Keizer buffering constant
 		    >) { deriveParameters->initialize2, initializeJunction->initialize2 };

NodeType VoltageJunctionPoint { produceInitialState->initialize1, produceVoltage->predictJunction };
NodeType CaConcentrationJunctionPoint { produceInitialState->initialize1, produceCaConcentration->predictJunction };

NodeType NaChannel { computeE_Na->initialize2, initializeNaChannels->initialize3, update->solveChannels };

NodeType KDRChannel_IO { computeE_KDR->initialize2, initializeKDRChannels->initialize3, update->solveChannels };

NodeType KCaChannel { computeE_K->initialize2, initializeKCaChannels->initialize3, update->solveChannels };

NodeType ChannelHCN (< E_HCN = {-43.0} >) {computeTadj->initialize2, initialize->initialize3, update->solveChannels };

NodeType CalChannel { initializeCalChannels->initialize3, update->solveChannels };

NodeType CahChannel { initialize->initialize3, update->solveChannels };
NodeType CaExtrusion (<tau_pump = 140, Ca_equil=0.1 >) { computeTadj->initialize2, initialize->initialize3, update->solveChannels };

NodeType CaConnexon { produceInitialState->initialize2, produceState->finish, computeState->solveChannels };

NodeType Connexon { produceInitialVoltage->initialize2, produceVoltage->finish, computeState->solveChannels };

NodeType AMPAReceptor (<
			    E=0,	  // mV
			    alpha=0.0011, // uM^-1 msec^-1
			    beta=0.19,    // msec^1
			    NTmax=180,     // 160-190 uM
			    Vp=2.0,       // mV
			    Kp=5.0        // mV
		      >) { computeTadj->initialize2, initializeAMPA->initialize3, updateAMPA->solveChannels };

NodeType GABAAReceptor (<
			    E=-80,	  // mV
			    alpha=0.005,  // uM^-1 msec^-1
			    beta=0.18,    // msec^1
			    NTmax=185,     // 185 um
			    Vp=2.0,       // mV
			    Kp=5.0        // mV
		      >) { computeTadj->initialize2, initializeGABAA->initialize3, updateGABAA->solveChannels };

NodeType PreSynapticPoint { produceInitialState->initialize2, produceState->finish };

ConstantType ExtracellularMedium;
ExtracellularMedium extracellularMedium<	Na=540.0, 	// uM
						K=20.0, 	// uM
					 	Ca=750.0,      // uM
						T=310.15>;      // degK  
ConstantType TimeStep;
//TimeStep timeStep<				deltaT=0.010    // msec - crash
TimeStep timeStep<				deltaT=0.005    // msec
		>;


Trigger UnsignedTrigger(string description, Service svc, string operator, int criterion, int delay, string phaseName);
Trigger CompositeTrigger(string description, Trigger triggerA, int critA, string operator, Trigger triggerB, int critB, int delay, string phaseName);


UnsignedTrigger currentOn("Iteration Trigger : == 100000", 
 			 ::Iteration, "==", 100000, 0, solveChannels );
//SA: Changed 10,200 to 12000
UnsignedTrigger currentMod("Iteration Trigger : == 102000", 
 			 ::Iteration, "==", 102000, 0, solveChannels );
//SA: Changed 10,400 to 14000
UnsignedTrigger currentOff("Iteration Trigger : == 104000", 
 			 ::Iteration, "==", 104000, 0, solveChannels );

UnsignedTrigger calciumOn("Iteration Trigger : == 1", 
 			 ::Iteration, "==", 1, 0, solveChannels );

UnsignedTrigger calciumOff("Iteration Trigger : == 50000", 
 			 ::Iteration, "==", 50000, 0, solveChannels );
	
UnsignedTrigger recOn("Iteration Trigger : !% 50", 
 			 ::Iteration, "!%", 50, 0, solveChannels );

VariableType PointCurrentSource { stimulate->solveChannels };
PointCurrentSource pointCurrentSource<>;

pointCurrentSource.setCurrent(<						
				I = 0.0                      	// pA
			     >) on currentOn;

pointCurrentSource.setCurrent(<						
				I = 2.0                      	// pA
			     >) on currentMod;

pointCurrentSource.setCurrent(<						
				I = 0.0            	        // pA
			     >) on currentOff;


VariableType PointCalciumSource { stimulate->solveChannels };
PointCalciumSource pointCalciumSource<>;

pointCalciumSource.setCaCurrent(<						
				I_Ca = 0.0                 	// pA_Ca
			     >) on calciumOn;

pointCalciumSource.setCaCurrent(<						
				I_Ca = 0.0                 	// pA_Ca
			     >) on calciumOff;

VariableType VoltageDisplay{ initialize->initialize1 };
VoltageDisplay voltageDisplay0<fileName=DAT0>;
VoltageDisplay voltageDisplay1<fileName=DAT1>;
VoltageDisplay voltageDisplay2<fileName=DAT2>;
//VoltageDisplay voltageDisplay3<fileName=DAT3>;
//VoltageDisplay voltageDisplay4<fileName=DAT4>;
//VoltageDisplay voltageDisplay5<fileName=DAT5>;
//VoltageDisplay voltageDisplay6<fileName=DAT6>;
//VoltageDisplay voltageDisplay7<fileName=DAT7>;
//VoltageDisplay voltageDisplay8<fileName=DAT8>;
//VoltageDisplay voltageDisplay9<fileName=DAT9>;
//VoltageDisplay voltageDisplay10<fileName=DAT10>;
//VoltageDisplay voltageDisplay11<fileName=DAT11>;
//VoltageDisplay voltageDisplay12<fileName=DAT12>;
//VoltageDisplay voltageDisplay13<fileName=DAT13>;
//VoltageDisplay voltageDisplay14<fileName=DAT14>;


//VariableType CalciumDisplay{ initialize->initialize1 };
//CalciumDisplay CalciumDisplay20<fileName=DAT20>;
//CalciumDisplay CalciumDisplay21<fileName=DAT21>;
//CalciumDisplay CalciumDisplay22<fileName=DAT22>;
//CalciumDisplay CalciumDisplay23<fileName=DAT23>;
//CalciumDisplay CalciumDisplay24<fileName=DAT24>;
//CalciumDisplay CalciumDisplay25<fileName=DAT25>;
//CalciumDisplay CalciumDisplay26<fileName=DAT26>;
//CalciumDisplay CalciumDisplay27<fileName=DAT27>;
//CalciumDisplay CalciumDisplay28<fileName=DAT28>;
//CalciumDisplay CalciumDisplay29<fileName=DAT29>;
//CalciumDisplay CalciumDisplay30<fileName=DAT30>;
//CalciumDisplay CalciumDisplay31<fileName=DAT31>;
//CalciumDisplay CalciumDisplay32<fileName=DAT32>;
//CalciumDisplay CalciumDisplay33<fileName=DAT33>;
//CalciumDisplay CalciumDisplay34<fileName=DAT34>;


voltageDisplay0.dataCollection(<>) on recOn;
voltageDisplay1.dataCollection(<>) on recOn;
voltageDisplay2.dataCollection(<>) on recOn;
//voltageDisplay3.dataCollection(<>) on recOn;
//voltageDisplay4.dataCollection(<>) on recOn;
//voltageDisplay5.dataCollection(<>) on recOn;
//voltageDisplay6.dataCollection(<>) on recOn;
//voltageDisplay7.dataCollection(<>) on recOn;
//voltageDisplay8.dataCollection(<>) on recOn;
//voltageDisplay9.dataCollection(<>) on recOn;
//voltageDisplay10.dataCollection(<>) on recOn;
//voltageDisplay11.dataCollection(<>) on recOn;
//voltageDisplay12.dataCollection(<>) on recOn;
//voltageDisplay13.dataCollection(<>) on recOn;
//voltageDisplay14.dataCollection(<>) on recOn;


//CalciumDisplay20.dataCollection(<>) on recOn;
//CalciumDisplay21.dataCollection(<>) on recOn;
//CalciumDisplay22.dataCollection(<>) on recOn;
//CalciumDisplay23.dataCollection(<>) on recOn;
//CalciumDisplay24.dataCollection(<>) on recOn;
//CalciumDisplay25.dataCollection(<>) on recOn;
//CalciumDisplay26.dataCollection(<>) on recOn;
//CalciumDisplay27.dataCollection(<>) on recOn;
//CalciumDisplay28.dataCollection(<>) on recOn;
//CalciumDisplay29.dataCollection(<>) on recOn;
//CalciumDisplay30.dataCollection(<>) on recOn;
//CalciumDisplay31.dataCollection(<>) on recOn;
//CalciumDisplay32.dataCollection(<>) on recOn;
//CalciumDisplay33.dataCollection(<>) on recOn;
//CalciumDisplay34.dataCollection(<>) on recOn;


Grid Tissue
{
   Dimension( _X_ , _Y_ , _Z_ );

   Layer(branches, HodgkinHuxleyVoltage, tissueFunctor("Layout", <nodekind="CompartmentVariables[Voltage]">), <nodekind="CompartmentVariables[Voltage]">, tissueGM);
   Layer(branches_Ca, CaConcentration, tissueFunctor("Layout", <nodekind="CompartmentVariables[Calcium]">), <nodekind="CompartmentVariables[Calcium]">, tissueGM);
   Layer(endPoints, VoltageEndPoint, tissueFunctor("Layout", <nodekind="EndPoints[Voltage]">), <nodekind="EndPoints[Voltage]">, tissueGM);
   Layer(endPoints_Ca, CaConcentrationEndPoint, tissueFunctor("Layout", <nodekind="EndPoints[Calcium]">), <nodekind="EndPoints[Calcium]">, tissueGM);





#if MAX_COMPUTE_ORDER>0
   Layer(fwdSolvePoints1, ForwardSolvePoint1, tissueFunctor("Layout", <nodekind="ForwardSolvePoints[Voltage][1]">), <nodekind="ForwardSolvePoints[Voltage][1]">, tissueGM);
   Layer(fwdSolvePoints1_Ca, ForwardSolvePoint1, tissueFunctor("Layout", <nodekind="ForwardSolvePoints[Calcium][1]">), <nodekind="ForwardSolvePoints[Calcium][1]">, tissueGM);
   Layer(bwdSolvePoints0, BackwardSolvePoint0, tissueFunctor("Layout", <nodekind="BackwardSolvePoints[Voltage][0]">), <nodekind="BackwardSolvePoints[Voltage][0]">, tissueGM);
   Layer(bwdSolvePoints0_Ca, BackwardSolvePoint0, tissueFunctor("Layout", <nodekind="BackwardSolvePoints[Calcium][0]">), <nodekind="BackwardSolvePoints[Calcium][0]">, tissueGM);
#endif
#if MAX_COMPUTE_ORDER>1
   Layer(fwdSolvePoints2, ForwardSolvePoint2, tissueFunctor("Layout", <nodekind="ForwardSolvePoints[Voltage][2]">), <nodekind="ForwardSolvePoints[Voltage][2]">, tissueGM);
   Layer(fwdSolvePoints2_Ca, ForwardSolvePoint2, tissueFunctor("Layout", <nodekind="ForwardSolvePoints[Calcium][2]">), <nodekind="ForwardSolvePoints[Calcium][2]">, tissueGM);
   Layer(bwdSolvePoints1, BackwardSolvePoint1, tissueFunctor("Layout", <nodekind="BackwardSolvePoints[Voltage][1]">), <nodekind="BackwardSolvePoints[Voltage][1]">, tissueGM);
   Layer(bwdSolvePoints1_Ca, BackwardSolvePoint1, tissueFunctor("Layout", <nodekind="BackwardSolvePoints[Calcium][1]">), <nodekind="BackwardSolvePoints[Calcium][1]">, tissueGM);
#endif
#if MAX_COMPUTE_ORDER>2
   Layer(fwdSolvePoints3, ForwardSolvePoint3, tissueFunctor("Layout", <nodekind="ForwardSolvePoints[Voltage][3]">), <nodekind="ForwardSolvePoints[Voltage][3]">, tissueGM);
   Layer(fwdSolvePoints3_Ca, ForwardSolvePoint3, tissueFunctor("Layout", <nodekind="ForwardSolvePoints[Calcium][3]">), <nodekind="ForwardSolvePoints[Calcium][3]">, tissueGM);
   Layer(bwdSolvePoints2, BackwardSolvePoint2, tissueFunctor("Layout", <nodekind="BackwardSolvePoints[Voltage][2]">), <nodekind="BackwardSolvePoints[Voltage][2]">, tissueGM);
   Layer(bwdSolvePoints2_Ca, BackwardSolvePoint2, tissueFunctor("Layout", <nodekind="BackwardSolvePoints[Calcium][2]">), <nodekind="BackwardSolvePoints[Calcium][2]">, tissueGM);
#endif
#if MAX_COMPUTE_ORDER>3
   Layer(fwdSolvePoints4, ForwardSolvePoint4, tissueFunctor("Layout", <nodekind="ForwardSolvePoints[Voltage][4]">), <nodekind="ForwardSolvePoints[Voltage][4]">, tissueGM);
   Layer(fwdSolvePoints4_Ca, ForwardSolvePoint4, tissueFunctor("Layout", <nodekind="ForwardSolvePoints[Calcium][4]">), <nodekind="ForwardSolvePoints[Calcium][4]">, tissueGM);
   Layer(bwdSolvePoints3, BackwardSolvePoint3, tissueFunctor("Layout", <nodekind="BackwardSolvePoints[Voltage][3]">), <nodekind="BackwardSolvePoints[Voltage][3]">, tissueGM);
   Layer(bwdSolvePoints3_Ca, BackwardSolvePoint3, tissueFunctor("Layout", <nodekind="BackwardSolvePoints[Calcium][3]">), <nodekind="BackwardSolvePoints[Calcium][3]">, tissueGM);
#endif
#if MAX_COMPUTE_ORDER>4
   Layer(fwdSolvePoints5, ForwardSolvePoint5, tissueFunctor("Layout", <nodekind="ForwardSolvePoints[Voltage][5]">), <nodekind="ForwardSolvePoints[Voltage][5]">, tissueGM);
   Layer(fwdSolvePoints5_Ca, ForwardSolvePoint5, tissueFunctor("Layout", <nodekind="ForwardSolvePoints[Calcium][5]">), <nodekind="ForwardSolvePoints[Calcium][5]">, tissueGM);
   Layer(bwdSolvePoints4, BackwardSolvePoint4, tissueFunctor("Layout", <nodekind="BackwardSolvePoints[Voltage][4]">), <nodekind="BackwardSolvePoints[Voltage][4]">, tissueGM);
   Layer(bwdSolvePoints4_Ca, BackwardSolvePoint4, tissueFunctor("Layout", <nodekind="BackwardSolvePoints[Calcium][4]">), <nodekind="BackwardSolvePoints[Calcium][4]">, tissueGM);
#endif
#if MAX_COMPUTE_ORDER>5
   Layer(fwdSolvePoints6, ForwardSolvePoint6, tissueFunctor("Layout", <nodekind="ForwardSolvePoints[Voltage][6]">), <nodekind="ForwardSolvePoints[Voltage][6]">, tissueGM);
   Layer(fwdSolvePoints6_Ca, ForwardSolvePoint6, tissueFunctor("Layout", <nodekind="ForwardSolvePoints[Calcium][6]">), <nodekind="ForwardSolvePoints[Calcium][6]">, tissueGM);
   Layer(bwdSolvePoints5, BackwardSolvePoint5, tissueFunctor("Layout", <nodekind="BackwardSolvePoints[Voltage][5]">), <nodekind="BackwardSolvePoints[Voltage][5]">, tissueGM);
   Layer(bwdSolvePoints5_Ca, BackwardSolvePoint5, tissueFunctor("Layout", <nodekind="BackwardSolvePoints[Calcium][5]">), <nodekind="BackwardSolvePoints[Calcium][5]">, tissueGM);
#endif
#if MAX_COMPUTE_ORDER>6
   Layer(fwdSolvePoints7, ForwardSolvePoint7, tissueFunctor("Layout", <nodekind="ForwardSolvePoints[Voltage][7]">), <nodekind="ForwardSolvePoints[Voltage][7]">, tissueGM);
   Layer(fwdSolvePoints7_Ca, ForwardSolvePoint7, tissueFunctor("Layout", <nodekind="ForwardSolvePoints[Calcium][7]">), <nodekind="ForwardSolvePoints[Calcium][7]">, tissueGM);
   Layer(bwdSolvePoints6, BackwardSolvePoint6, tissueFunctor("Layout", <nodekind="BackwardSolvePoints[Voltage][6]">), <nodekind="BackwardSolvePoints[Voltage][6]">, tissueGM);
   Layer(bwdSolvePoints6_Ca, BackwardSolvePoint6, tissueFunctor("Layout", <nodekind="BackwardSolvePoints[Calcium][6]">), <nodekind="BackwardSolvePoints[Calcium][6]">, tissueGM);
#endif

   Layer(junctions, HodgkinHuxleyVoltageJunction, tissueFunctor("Layout", <nodekind="Junctions[Voltage]">), <nodekind="Junctions[Voltage]">, tissueGM);
   Layer(junctions_Ca, CaConcentrationJunction, tissueFunctor("Layout", <nodekind="Junctions[Calcium]">), <nodekind="Junctions[Calcium]">, tissueGM);

   Layer(junctionPoints, VoltageJunctionPoint, tissueFunctor("Layout", <nodekind="JunctionPoints[Voltage]">), <nodekind="JunctionPoints[Voltage]">, tissueGM);
   Layer(junctionPoints_Ca, CaConcentrationJunctionPoint, tissueFunctor("Layout", <nodekind="JunctionPoints[Calcium]">), <nodekind="JunctionPoints[Calcium]">, tissueGM);

 Layer(PreSynapticPoints, PreSynapticPoint, tissueFunctor("Layout", < nodekind="PreSynapticPoints[Voltage]" >), < nodekind="PreSynapticPoints[Voltage]" >, tissueGM);

 


   Layer(NaChannels, NaChannel, tissueFunctor("Layout", < nodekind="Channels[Na]" >), < nodekind="Channels[Na]" >, tissueGM);
   Layer(KDRChannels, KDRChannel_IO, tissueFunctor("Layout", < nodekind="Channels[KDR]" >), < nodekind="Channels[KDR]" >, tissueGM);
   Layer(KCaChannels, KCaChannel, tissueFunctor("Layout", < nodekind="Channels[KCa]" >), < nodekind="Channels[KCa]" >, tissueGM);
   Layer(HCNChannels, ChannelHCN, tissueFunctor("Layout", < nodekind="Channels[HCN]" >), < nodekind="Channels[HCN]" >, tissueGM);
   Layer(CalChannels, CalChannel, tissueFunctor("Layout", < nodekind="Channels[Cal]" >), < nodekind="Channels[Cal]" >, tissueGM);
   Layer(CahChannels, CahChannel, tissueFunctor("Layout", < nodekind="Channels[Cah]" >), < nodekind="Channels[Cah]" >, tissueGM);
   Layer(CaExChannels, CaExtrusion, tissueFunctor("Layout", < nodekind="Channels[CaEx]" >), < nodekind="Channels[CaEx]" >, tissueGM);

  

   Layer(AMPASynapses, AMPAReceptor, tissueFunctor("Layout", < nodekind="ChemicalSynapses[AMPA]" >), < nodekind="ChemicalSynapses[AMPA]" >, tissueGM);
   Layer(GABAASynapses, GABAAReceptor, tissueFunctor("Layout", < nodekind="ChemicalSynapses[GABAA]" >), < nodekind="ChemicalSynapses[GABAA]" >, tissueGM);
  

Layer(DendroDendriticGapJunctions,Connexon, tissueFunctor("Layout", < nodekind="ElectricalSynapses[DenDenGap]" >), < nodekind="ElectricalSynapses[DenDenGap]" >, tissueGM);


   InitNodes ( .[].Layer(branches), tissueFunctor("NodeInit", <
									compartmentalize = {"Vnew", 
											    "Vcur", 
											    "Aii", 
											    "Aim", 
											    "Aip", 
											    "RHS"
											   },
									Vnew = {-64.1235},
									Cm=V_CM,		// pF/um^2
									gLeak=V_GLEAK 		// nS/um^2, Hines used 0.0003 S/cm^2
								      > ) );

   InitNodes ( .[].Layer(branches_Ca), tissueFunctor("NodeInit", <
									compartmentalize = {"Ca_new", 
											    "Ca_cur", 
											    "currentToConc",
											    "Aii",
											    "Aim", 
											    "Aip", 
											    "RHS"
											   },
									Ca_new = {0.1}
								//	CaClearance=CA_CLEARANCE    // s^-1
								      > ) );

   InitNodes ( .[].Layer(endPoints), tissueFunctor("NodeInit", <> ) );
   InitNodes ( .[].Layer(endPoints_Ca), tissueFunctor("NodeInit", <> ) );
#if MAX_COMPUTE_ORDER>0
   InitNodes ( .[].Layer(fwdSolvePoints1), tissueFunctor("NodeInit", <> ) );
   InitNodes ( .[].Layer(fwdSolvePoints1_Ca), tissueFunctor("NodeInit", <> ) );
   InitNodes ( .[].Layer(bwdSolvePoints0), tissueFunctor("NodeInit", <> ) );
   InitNodes ( .[].Layer(bwdSolvePoints0_Ca), tissueFunctor("NodeInit", <> ) );
#endif
#if MAX_COMPUTE_ORDER>1
   InitNodes ( .[].Layer(fwdSolvePoints2), tissueFunctor("NodeInit", <> ) );
   InitNodes ( .[].Layer(fwdSolvePoints2_Ca), tissueFunctor("NodeInit", <> ) );
   InitNodes ( .[].Layer(bwdSolvePoints1), tissueFunctor("NodeInit", <> ) );
   InitNodes ( .[].Layer(bwdSolvePoints1_Ca), tissueFunctor("NodeInit", <> ) );
#endif
#if MAX_COMPUTE_ORDER>2
   InitNodes ( .[].Layer(fwdSolvePoints3), tissueFunctor("NodeInit", <> ) );
   InitNodes ( .[].Layer(fwdSolvePoints3_Ca), tissueFunctor("NodeInit", <> ) );
   InitNodes ( .[].Layer(bwdSolvePoints2), tissueFunctor("NodeInit", <> ) );
   InitNodes ( .[].Layer(bwdSolvePoints2_Ca), tissueFunctor("NodeInit", <> ) );
#endif
#if MAX_COMPUTE_ORDER>3
   InitNodes ( .[].Layer(fwdSolvePoints4), tissueFunctor("NodeInit", <> ) );
   InitNodes ( .[].Layer(fwdSolvePoints4_Ca), tissueFunctor("NodeInit", <> ) );
   InitNodes ( .[].Layer(bwdSolvePoints3), tissueFunctor("NodeInit", <> ) );
   InitNodes ( .[].Layer(bwdSolvePoints3_Ca), tissueFunctor("NodeInit", <> ) );
#endif
#if MAX_COMPUTE_ORDER>4
   InitNodes ( .[].Layer(fwdSolvePoints5), tissueFunctor("NodeInit", <> ) );
   InitNodes ( .[].Layer(fwdSolvePoints5_Ca), tissueFunctor("NodeInit", <> ) );
   InitNodes ( .[].Layer(bwdSolvePoints4), tissueFunctor("NodeInit", <> ) );
   InitNodes ( .[].Layer(bwdSolvePoints4_Ca), tissueFunctor("NodeInit", <> ) );
#endif
#if MAX_COMPUTE_ORDER>5
   InitNodes ( .[].Layer(fwdSolvePoints6), tissueFunctor("NodeInit", <> ) );
   InitNodes ( .[].Layer(fwdSolvePoints6_Ca), tissueFunctor("NodeInit", <> ) );
   InitNodes ( .[].Layer(bwdSolvePoints5), tissueFunctor("NodeInit", <> ) );
   InitNodes ( .[].Layer(bwdSolvePoints5_Ca), tissueFunctor("NodeInit", <> ) );
#endif
#if MAX_COMPUTE_ORDER>6
   InitNodes ( .[].Layer(fwdSolvePoints7), tissueFunctor("NodeInit", <> ) );
   InitNodes ( .[].Layer(fwdSolvePoints7_Ca), tissueFunctor("NodeInit", <> ) );
   InitNodes ( .[].Layer(bwdSolvePoints6), tissueFunctor("NodeInit", <> ) );
   InitNodes ( .[].Layer(bwdSolvePoints6_Ca), tissueFunctor("NodeInit", <> ) );
#endif
   InitNodes ( .[].Layer(junctions), tissueFunctor("NodeInit", < Vnew = {-64.3346},
								 Cm=V_CM,		// pF/um^2
								 gLeak=V_GLEAK 		// nS/um^2, Hines used 0.0003 S/cm^2
								> ) );
   InitNodes ( .[].Layer(junctions_Ca), tissueFunctor("NodeInit", < Ca_new = {0.1} > ) );
   InitNodes ( .[].Layer(junctionPoints), tissueFunctor("NodeInit", <> ) );
   InitNodes ( .[].Layer(junctionPoints_Ca), tissueFunctor("NodeInit", <> ) );


 InitNodes ( .[].Layer(PreSynapticPoints), tissueFunctor("NodeInit", <> ) );


   InitNodes ( .[].Layer(NaChannels), tissueFunctor("NodeInit", <
									compartmentalize = { "gbar" }
								 > ) );	


   InitNodes ( .[].Layer(KDRChannels), tissueFunctor("NodeInit", <
									compartmentalize = {  "gbar" }
								 > ) );

   InitNodes ( .[].Layer(KCaChannels), tissueFunctor("NodeInit", <
									compartmentalize = {  "gbar" }
//									, gbar = {3.5}
								 > ) );

   InitNodes ( .[].Layer(HCNChannels), tissueFunctor("NodeInit", <
									compartmentalize = {  "gbar" },
									gbar = {0.85}
								 > ) );

   InitNodes ( .[].Layer(CalChannels), tissueFunctor("NodeInit", <
									compartmentalize = { "gbar" },
									gbar = {1.5}
								 > ) );

   InitNodes ( .[].Layer(CahChannels), tissueFunctor("NodeInit", <
									compartmentalize = { "gbar" }
//									, gbar = {0.4}
								 > ) );

   InitNodes ( .[].Layer(CaExChannels), tissueFunctor("NodeInit", <
									compartmentalize = { "tau" }
									, tau = {40.0}
								 > ) );
   InitNodes ( .[].Layer(DendroDendriticGapJunctions), tissueFunctor("NodeInit", <
									I = 0,
									g = 0.5
								  > ) );
   InitNodes ( .[].Layer(AMPASynapses), tissueFunctor("NodeInit", <
									gbar = 0.1
								  > ) );
   InitNodes ( .[].Layer(GABAASynapses), tissueFunctor("NodeInit", <
									gbar = 0.1
								  > ) );	
   
  

   polyConnect(timeStep, .[].Layer(branches, junctions, branches_Ca, junctions_Ca, NaChannels, KDRChannels, KCaChannels, HCNChannels, CalChannels, CahChannels, CaExChannels), <>, <identifier="dt">);
   polyConnect(extracellularMedium, .[].Layer(NaChannels, KDRChannels, KCaChannels, HCNChannels, CalChannels, CahChannels, CaExChannels, DendroDendriticGapJunctions
   ), <>, <identifier="EC">);
   polyConnect(timeStep, .[].Layer(AMPASynapses), <>, <identifier="dt">);
   polyConnect(timeStep, .[].Layer(GABAASynapses), <>, <identifier="dt">);

   tissueFunctor("Connect", <> );	

   polyConnect(pointCurrentSource, tissueFunctor("Probe", <CATEGORY="JUNCTION", TYPE="Voltage", BRANCHTYPE=1, NEURON_INDEX=0>), <>, <identifier="stimulation">);
   polyConnect(pointCurrentSource, tissueFunctor("Probe", <CATEGORY="BRANCH", TYPE="Voltage", BRANCHTYPE=3, NEURON_INDEX=0>), <>, <identifier="stimulation", idx=-1>);
   polyConnect(pointCalciumSource, tissueFunctor("Probe", <CATEGORY="JUNCTION", TYPE="Calcium", BRANCHTYPE=1, NEURON_INDEX=0>), <>, <identifier="stimulation">);

   polyConnect(timeStep, voltageDisplay0, <>, <>);
   polyConnect(timeStep, voltageDisplay1, <>, <>);
   polyConnect(timeStep, voltageDisplay2, <>, <>);
   //polyConnect(timeStep, voltageDisplay3, <>, <>);
   //polyConnect(timeStep, voltageDisplay4, <>, <>);
   //polyConnect(timeStep, voltageDisplay5, <>, <>);
   //polyConnect(timeStep, voltageDisplay6, <>, <>);
   //polyConnect(timeStep, voltageDisplay7, <>, <>);
   //polyConnect(timeStep, voltageDisplay8, <>, <>);	
   //polyConnect(timeStep, voltageDisplay9, <>, <>);	
   //polyConnect(timeStep, voltageDisplay10, <>, <>);	
   //polyConnect(timeStep, voltageDisplay11, <>, <>);	
   //polyConnect(timeStep, voltageDisplay12, <>, <>);	
   //polyConnect(timeStep, voltageDisplay13, <>, <>);	
   //polyConnect(timeStep, voltageDisplay14, <>, <>);	

   //polyConnect(timeStep, CalciumDisplay20, <>, <>);
   //polyConnect(timeStep, CalciumDisplay21, <>, <>);
   //polyConnect(timeStep, CalciumDisplay22, <>, <>);
   //polyConnect(timeStep, CalciumDisplay23, <>, <>);
   //polyConnect(timeStep, CalciumDisplay24, <>, <>);
   //polyConnect(timeStep, CalciumDisplay25, <>, <>);
   //polyConnect(timeStep, CalciumDisplay26, <>, <>);
   //polyConnect(timeStep, CalciumDisplay27, <>, <>);
   //polyConnect(timeStep, CalciumDisplay28, <>, <>);
   //polyConnect(timeStep, CalciumDisplay29, <>, <>);
   //polyConnect(timeStep, CalciumDisplay30, <>, <>);
   //polyConnect(timeStep, CalciumDisplay31, <>, <>);
   //polyConnect(timeStep, CalciumDisplay32, <>, <>);
   //polyConnect(timeStep, CalciumDisplay33, <>, <>);
   //polyConnect(timeStep, CalciumDisplay34, <>, <>);

   polyConnect( tissueFunctor("Probe", <CATEGORY="BRANCH", TYPE="Voltage",  BRANCHTYPE=2, NEURON_INDEX=0>), voltageDisplay1, <>, <>);
   polyConnect( tissueFunctor("Probe", <CATEGORY="BRANCH", TYPE="Voltage", BRANCHTYPE=3, NEURON_INDEX=0>), voltageDisplay2, <>, <>);
//   polyConnect( tissueFunctor("Probe", <CATEGORY="JUNCTION", TYPE="Voltage", BRANCHORDER=0, NEURON_INDEX=1>), voltageDisplay3, <>, <>);
//   polyConnect( tissueFunctor("Probe", <CATEGORY="BRANCH", TYPE="Voltage", BRANCHORDER=3, BRANCHTYPE=1, NEURON_INDEX=1>), voltageDisplay4, <>, <>);
//   polyConnect( tissueFunctor("Probe", <CATEGORY="BRANCH", TYPE="Voltage", BRANCHORDER=3, BRANCHTYPE=2, NEURON_INDEX=1>), voltageDisplay5, <>, <>);
//   polyConnect( tissueFunctor("Probe", <CATEGORY="JUNCTION", TYPE="Voltage", BRANCHORDER=0, NEURON_INDEX=2>), voltageDisplay6, <>, <>);
//   polyConnect( tissueFunctor("Probe", <CATEGORY="BRANCH", TYPE="Voltage", BRANCHORDER=3, BRANCHTYPE=1, NEURON_INDEX=2>), voltageDisplay7, <>, <>);
//   polyConnect( tissueFunctor("Probe", <CATEGORY="BRANCH", TYPE="Voltage", BRANCHORDER=3, BRANCHTYPE=2, NEURON_INDEX=2>), voltageDisplay8, <>, <>);
//   polyConnect( tissueFunctor("Probe", <CATEGORY="JUNCTION", TYPE="Voltage", BRANCHORDER=0, NEURON_INDEX=3>), voltageDisplay9, <>, <>);
//   polyConnect( tissueFunctor("Probe", <CATEGORY="BRANCH", TYPE="Voltage", BRANCHORDER=3, BRANCHTYPE=1, NEURON_INDEX=3>), voltageDisplay10, <>, <>);
//   polyConnect( tissueFunctor("Probe", <CATEGORY="BRANCH", TYPE="Voltage", BRANCHORDER=3, BRANCHTYPE=2, NEURON_INDEX=3>), voltageDisplay11, <>, <>);
//   polyConnect( tissueFunctor("Probe", <CATEGORY="JUNCTION", TYPE="Voltage", BRANCHORDER=0, NEURON_INDEX=4>), voltageDisplay12, <>, <>);
//   polyConnect( tissueFunctor("Probe", <CATEGORY="BRANCH", TYPE="Voltage", BRANCHORDER=3, BRANCHTYPE=1, NEURON_INDEX=4>), voltageDisplay13, <>, <>);
//   polyConnect( tissueFunctor("Probe", <CATEGORY="BRANCH", TYPE="Voltage", BRANCHORDER=3, BRANCHTYPE=2, NEURON_INDEX=4>), voltageDisplay14, <>, <>);
//
//
//   polyConnect( tissueFunctor("Probe", <CATEGORY="JUNCTION", TYPE="Calcium", BRANCHORDER=0, NEURON_INDEX=0>), CalciumDisplay20, <>, <>);
//   polyConnect( tissueFunctor("Probe", <CATEGORY="BRANCH", TYPE="Calcium", BRANCHORDER=3, BRANCHTYPE=1, NEURON_INDEX=0>), CalciumDisplay21, <>, <>);
//   polyConnect( tissueFunctor("Probe", <CATEGORY="BRANCH", TYPE="Calcium", BRANCHORDER=3, BRANCHTYPE=2, NEURON_INDEX=0>), CalciumDisplay22, <>, <>);
//   polyConnect( tissueFunctor("Probe", <CATEGORY="JUNCTION", TYPE="Calcium", BRANCHORDER=0, NEURON_INDEX=1>), CalciumDisplay23, <>, <>);
//   polyConnect( tissueFunctor("Probe", <CATEGORY="BRANCH", TYPE="Calcium", BRANCHORDER=3, BRANCHTYPE=1, NEURON_INDEX=1>), CalciumDisplay24, <>, <>);
//   polyConnect( tissueFunctor("Probe", <CATEGORY="BRANCH", TYPE="Calcium", BRANCHORDER=3, BRANCHTYPE=2, NEURON_INDEX=1>), CalciumDisplay25, <>, <>);
//   polyConnect( tissueFunctor("Probe", <CATEGORY="JUNCTION", TYPE="Calcium", BRANCHORDER=0, NEURON_INDEX=2>), CalciumDisplay26, <>, <>);
//   polyConnect( tissueFunctor("Probe", <CATEGORY="BRANCH", TYPE="Calcium", BRANCHORDER=3, BRANCHTYPE=1, NEURON_INDEX=2>), CalciumDisplay27, <>, <>);
//   polyConnect( tissueFunctor("Probe", <CATEGORY="BRANCH", TYPE="Calcium", BRANCHORDER=3, BRANCHTYPE=2, NEURON_INDEX=2>), CalciumDisplay28, <>, <>);
//   polyConnect( tissueFunctor("Probe", <CATEGORY="JUNCTION", TYPE="Calcium", BRANCHORDER=0, NEURON_INDEX=3>), CalciumDisplay29, <>, <>);
//   polyConnect( tissueFunctor("Probe", <CATEGORY="BRANCH", TYPE="Calcium", BRANCHORDER=3, BRANCHTYPE=1, NEURON_INDEX=3>), CalciumDisplay30, <>, <>);
//   polyConnect( tissueFunctor("Probe", <CATEGORY="BRANCH", TYPE="Calcium", BRANCHORDER=3, BRANCHTYPE=2, NEURON_INDEX=3>), CalciumDisplay31, <>, <>);
//   polyConnect( tissueFunctor("Probe", <CATEGORY="JUNCTION", TYPE="Calcium", BRANCHORDER=0, NEURON_INDEX=4>), CalciumDisplay32, <>, <>);
//   polyConnect( tissueFunctor("Probe", <CATEGORY="BRANCH", TYPE="Calcium", BRANCHORDER=3, BRANCHTYPE=1, NEURON_INDEX=4>), CalciumDisplay33, <>, <>);
//   polyConnect( tissueFunctor("Probe", <CATEGORY="BRANCH", TYPE="Calcium", BRANCHORDER=3, BRANCHTYPE=2, NEURON_INDEX=4>), CalciumDisplay34, <>, <>);
};

Tissue tissue;

Grid Adaptor
{
   Dimension( _X_ , _Y_ , _Z_ );
   Layer(DendroDendriticGapJunctionConnexons0, Connexon, tissueFunctor("Layout", <PROBED="pr0", N=6, PRMASK0>), <>, tissueGM);
   Layer(DendroDendriticGapJunctionConnexons1, Connexon, tissueFunctor("Layout", <PROBED="pr1", N=6, PRMASK1>), <>, tissueGM);

   


   BindName cnnxn ("I", 0, "g", 0.5);
   NdplNodeInit Mcnnxn(cnnxn);

   InitNodes ( .[].Layer(DendroDendriticGapJunctionConnexons0), Mcnnxn );
   InitNodes ( .[].Layer(DendroDendriticGapJunctionConnexons1), Mcnnxn );

   BindName cpt2cnnxn("idx", -1,
   	 	      "identifier", "compartment[Voltage]");
   NdplInAttrInit Mcpt2cnnxn(cpt2cnnxn);

   BindName cnnxn2cpt("idx", -1,
   	 	      "identifier", "electricalSynapse[Voltage]");
   NdplInAttrInit Mcnnxn2cpt(cnnxn2cpt);

   BindName cnnxn2cnnxn("identifier", "connexon[Voltage]");
   NdplInAttrInit Mcnnxn2cnnxn(cnnxn2cnnxn);

   zipper(tissueFunctor("Probe", <PROBED="pr0", PRMASK0>), .[].Layer(DendroDendriticGapJunctionConnexons0), outAttrDef, Mcpt2cnnxn, "ids0");
   zipper(tissueFunctor("Probe", <PROBED="pr1", PRMASK1>), .[].Layer(DendroDendriticGapJunctionConnexons1), outAttrDef, Mcpt2cnnxn, "ids1");

   zipper(.[].Layer(DendroDendriticGapJunctionConnexons0), tissueFunctor("Probe", <PROBED="pr0", PRMASK0>), outAttrDef, Mcnnxn2cpt, "ids0");
   zipper(.[].Layer(DendroDendriticGapJunctionConnexons1), tissueFunctor("Probe", <PROBED="pr1", PRMASK1>), outAttrDef, Mcnnxn2cpt, "ids1");

   connectNodeSets(.[].Layer(DendroDendriticGapJunctionConnexons0), .[].Layer(DendroDendriticGapJunctionConnexons1), isoSampler, outAttrDef, Mcnnxn2cnnxn);
   connectNodeSets(.[].Layer(DendroDendriticGapJunctionConnexons1), .[].Layer(DendroDendriticGapJunctionConnexons0), isoSampler, outAttrDef, Mcnnxn2cnnxn);
};

//Adaptor adaptor;

// DCA directives here

UnsignedTrigger endTrig("Iteration Trigger to end or stop", 
			 ::Iteration, "==", 300000, 0, correctJunction);
Stop on endTrig;
