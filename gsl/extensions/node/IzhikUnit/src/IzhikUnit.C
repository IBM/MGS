#include "Lens.h"
#include "IzhikUnit.h"
#include "CG_IzhikUnit.h"
#include "rndm.h"
<<<<<<< HEAD
#include "NumInt.h"
=======
//#include "NumIntNoPhase.h"
>>>>>>> origin/team-A

#define SHD getSharedMembers()
#define ITER getSimulation().getIteration()
#define TIME ITER*getSharedMembers().deltaT
#define RANK getSimulation().getRank()
#define NUMVARS 3

template <typename T>
bool Threshold(T val, T thresh)
{
  return val>thresh;
}

template <typename T>
T Sigmoid(const T & val, const T & V_inf, const T & k_inf) 
{return 1.0/(1.0 + exp((V_inf - val)/k_inf));}


void IzhikUnit::initialize(RNG& rng) 
{

  /*
  drivinps.increaseSizeTo(SHD.statenum);

  ShallowArray<double>::iterator iter = drivinps.begin(), end=drivinps.end();
  for (; iter!=end; ++iter) (*iter) =  drandom(inplo,inphi,rng);

  drivinp = & drivinps[0];
  */

  nodeVars.increaseSizeTo(NUMVARS);
  //nodeVars = 0;
  initializeIterator(NUMVARS,SHD.deltaT);
  nodeVars[0] = V_init;
  nodeVars[1] = g_init;
  nodeVars[2] = n_init;


  /*  
  std::ofstream ofs;
  ofs.open ("weights.dat", std::ofstream::out | std::ofstream::app);
  outputWeights(ofs);
  ofs.close();
  */

  /*
  std::ofstream ofs;
  ofs.open ("inps.dat", std::ofstream::out | std::ofstream::app);
  outputDrivInp(ofs);
  ofs.close();
  */

}

void IzhikUnit::update(RNG& rng) 
{
  callIterate(nodeVars);
}

void IzhikUnit::updateOutputs(RNG& rng) 
{
  const float v1 = (float) nodeVars[0];
  if (v1 >= SHD.spikethresh && V_out < SHD.spikethresh) spike = true;
  else spike = false;
  V_out = v1;
  g_out = nodeVars[1];

}

IzhikUnit::~IzhikUnit() 
{
}

void IzhikUnit::derivs(const ShallowArray< double > & x, ShallowArray< double > & dx)
{
  
  const double & V = x[0];
  double & dV = dx[0];
  const double & g = x[1];
  double & dg = dx[1];
  const double & n = x[2];
  double & dn = dx[2];

  
  dg = (Threshold<const double &>(V,SHD.spikethresh) - g)/SHD.tau_g;

  ShallowArray<Input>::iterator iter, end=MSNNetInps.end();
  double drive=0;
  for (iter=MSNNetInps.begin(); iter!=end; ++iter) {
    drive += *(iter->input)*iter->weight;
  }

  //dV = 0;
  dV = drive*(V - SHD.Vconnectsyn);
  dV += SHD.gl*(V-SHD.El);
  dV += SHD.gk*(V-SHD.Ek)*n;
  dV += SHD.gna*(V-SHD.Ena)*Sigmoid<double>(V,SHD.mv12,SHD.mk);
  dV += (*drivinp);
  dV /=SHD.C;


  dn = (Sigmoid<double>(V,SHD.nv12,SHD.nk) - n)/SHD.tau_n;

  //dn = 0;
  //dg = 0;
  //dV = 0;


}


void IzhikUnit::setIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_IzhikUnitInAttrPSet* CG_inAttrPset, CG_IzhikUnitOutAttrPSet* CG_outAttrPset) 
{

  MSNNetInps[MSNNetInps.size()-1].row = getGlobalIndex()+1; // +1 is for Matlab
  MSNNetInps[MSNNetInps.size()-1].col = CG_node->getGlobalIndex()+1; 

}


void IzhikUnit::outputWeights(std::ofstream& fs)
{
  ShallowArray<Input>::iterator iter, end=MSNNetInps.end();

  for (iter=MSNNetInps.begin(); iter!=end; ++iter)
    fs<<iter->row<<" "<<iter->col<<" "<<iter->weight<<std::endl;
}


/*
void IzhikUnit::outputDrivInp(std::ofstream& fs)
{
  fs << (*drivinp) << std::endl;
}
*/
