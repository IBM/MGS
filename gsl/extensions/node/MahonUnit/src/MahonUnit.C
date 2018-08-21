#include "Lens.h"
#include "MahonUnit.h"
#include "CG_MahonUnit.h"
#include "GridLayerData.h"
#include "rndm.h"
#include "NumInt.h"

#define SHD getSharedMembers()
#define ITER getSimulation().getIteration()
#define TIME ITER*getSharedMembers().deltaT
#define RANK getSimulation().getRank()
#define NUMVARS 12

template <typename T>
bool Threshold(T val, T thresh)
{
  return val>thresh;
}

template <typename T>
T sigmoid(const T & V, const T Vb, const T k) 
{
return 1.0/(1.0 + exp(-1.0*(V - Vb)/k));
}

template <typename T>
T pow(const T & val, const int & q)
{
  int i = 0;
  T retval = 1.0;
  while(i<q) {retval*=val;i++;}
  return val;
}


template <typename T>
T IonChannel(const T & V, const T & m, const T & h, const T g, const T Vb, 
	     const int p, const int q) 
{
  return g*pow(m,p)*pow(h,q)*(V-Vb);
}

template <typename T>
T IonChannel31(const T & V, const T & m, const T & h, const T g, const T Vb) 
{
  return g*m*m*m*h*(V-Vb);
}

template <typename T>
T IonChannel4(const T & V, const T & m, const T g, const T Vb) 
{
  return g*m*m*m*m*(V-Vb);
}

template <typename T>
T IonChannel(const T & V, const T & m, const T & h, const T g, const T Vb) 
{
  return g*m*h*(V-Vb);
}

template <typename T>
T IonChannel(const T & V, const T & m, const T g, const T Vb) 
{
  return g*m*(V-Vb);
}


template <typename T>
T IonChannel(const T & V, const T g, const T Vb) 
{
  return g*(V-Vb);
}


template <typename T>
T ratefcn(const T & x, const T xb, const T t) 
{
  return (xb - x)/t;
}


template <typename T>
T taufcn(const T & x, const T tau1, const T phi, const T sig0) 
{
  const T val1 = (x - phi)/sig0;
  return tau1/(exp(val1) + exp(val1*-1.0));
}


template <typename T>
T Ashtaufcn(const T & x)
{
  const T val1 = (x + 38.2)/28.0;
  return 1790.0 + 2930.0*exp(val1*val1*-1.0)*val1;
}





template <typename T> //wang buzaki 96 V shifted 7mv
T Kmalpha(const T & x)
{
  const T val1 = x + 27.0;
  return -0.01*val1/(exp(-0.1*val1) - 1.0);
}

template <typename T> //wang buzaki 96  V shifted 7mv
T Kmbeta(const T & x)
{
  return 0.125*exp((x+37.0)/-80.0);
}

template <typename T> //wang buzaki 96 V shifted 7mv
T Namalpha(const T & x)
{
  const T val1 = x + 28.0;
  return -0.1*val1/(exp(-0.1*val1) - 1.0);
}

template <typename T> //wang buzaki 96  V shifted 7mv
T Nambeta(const T & x)
{
  return 4.0*exp((x+53.0)/-18.0);
}

template <typename T> //wang buzaki 96 V shifted 7mv
T Nahalpha(const T & x)
{
  return 0.07*exp(-(x+51.0)/20.0);
}

template <typename T> //wang buzaki 96  V shifted 7mv
T Nahbeta(const T & x)
{
  return 1.0/(exp(-0.1*(x+21.0)) + 1.0);
}


template <typename T> //wang buzaki 96
T gatefcn(const T & x, const T alpha, const T beta, const T scale)
{
  return scale * (alpha*(1.0-x) - beta*x);
}


template <typename T> //wang buzaki 96
T gatefcnInstant(const T alpha, const T beta)
{
  return alpha / (alpha + beta);
}

template <typename T>
T Tadj(const T q10, const T cels, const T Etemp)
{
  return pow(q10, (cels-Etemp)/10.0);
}



template <typename T>
T TadjAdj(const T x, const T tadj)
{
  return x/tadj;
}

#define q10 2.5
#define CELSIUS 37
#define ETEMPKAF 22 //KAf, KAS, Krpm, Nap, 22
#define ETEMPNAS 21 
#define TADJPKAF 3.953
#define TADJPNAS 4.332


#define GLEAK 0.075
#define VLEAK -90.0 //-75.0 //(Mahon) //-90.0 (Gittis)
#define GNAS 0.11
#define VNAS 40.0
#define GNAP 0.02
#define VNAP 45.0
#define GKRP 0.42
#define VKRP -77.5
#define GAS 0.32
#define VAS -85.0
#define GAF 0.09
#define VAF -73.0
#define GKIR 0.15
#define VKIR -90.0
#define GKCHAN 6.0
#define VKCHAN -90.0
#define GNACHAN 35.0
#define VNACHAN 55.0

#define NASMTHE -16.0 
#define NASMK 9.4 
#define NASMTAU1 637.8
#define NASMPHI -33.5
#define NASMSIG0 26.3

#define NAPMTHE -47.8 
#define NAPMK 3.1 
#define NAPMTAU1 1.0

#define KRPMTHE -13.4 
#define KRPMK 12.1 
#define KRPMTAU1 206.2 
#define KRPMPHI -53.9 
#define KRPMSIG0 26.5

#define KRPHTHE -55.0 
#define KRPHK -19.0 

#define ASHTHE -78.8
#define ASHK -10.4

#define XASMTHE -25.6
#define XASMK 13.3
#define XASMTAU1 131.4
#define XASMPHI -37.4
#define XASMSIG0 27.3

#define AFMTHE -33.1 
#define AFMK 7.5
#define AFMTAU1 1.0

#define AFHTHE -70.4 
#define AFHK -7.6
#define AFHTAU1 25.0

#define KIRMTHE -100.0
#define KIRMK -10.0
#define KIRMTAU1 0.01

#define VCL -80.0
#define SYNA 2
#define SYNB 0.1

#define CAPAC 1.0

void MahonUnit::derivs(const ShallowArray< double > & x, ShallowArray< double > & dx)
{

  const double & V = x[0];
  double & dV = dx[0];
  const double & Nasm = x[1];
  double & dNasm = dx[1];
  const double & Napm = x[2];
  double & dNapm = dx[2];
  const double & Krpm = x[3];
  double & dKrpm = dx[3];
  const double & Krph = x[4];
  double & dKrph = dx[4];
  const double & Asm = x[5];
  double & dAsm = dx[5];
  const double & Ash = x[6];
  double & dAsh = dx[6];
  const double & Afm = x[7];
  double & dAfm = dx[7];
  const double & Afh = x[8];
  double & dAfh = dx[8];
  const double & Km = x[9];
  double & dKm = dx[9];
  //const double & Nam = x[11];
  //double & dNam = dx[11];
  const double & Nah = x[10];
  double & dNah = dx[10];

  //  const double & Kirm = x[11];
  //double & dKirm = dx[11];

  const double & g = x[11];
  double & dg = dx[11];



  ShallowArray<Input>::iterator iter, end=MSNNetInps.end();
  double drive=0;
  for (iter=MSNNetInps.begin(); iter!=end; ++iter) {
    drive += *(iter->input)*iter->weight;
  }
  
  dg = SYNA*((double) (V>0.0))*(1.0 - g) - synb*g;


  //dV = 0;

  dV = drive*(V - VCL);
  dV -= IonChannel<double>(V,GLEAK,VLEAK);

  dV -= IonChannel<double>(V,Nasm,GNAS,VNAS);
  dV -= IonChannel<double>(V,Napm,GNAP,VNAP);
  dV -= IonChannel<double>(V,Krpm,Krph,GKRP,VKRP);
  dV -= IonChannel<double>(V,Asm,Ash,GAS,VAS);
  dV -= IonChannel<double>(V,Afm,Afh,GAF,VAF);

  //dV -= IonChannel<double>(V,Kirm,GKIR,VKIR);

  dV -= IonChannel<double>(V,sigmoid<float>(V,KIRMTHE,KIRMK),GKIR,VKIR);

  dV -= IonChannel4<double>(V,Km,GKCHAN,VKCHAN);
  dV -= IonChannel31<double>
    (V,gatefcnInstant<double>(Namalpha<double>(V),Nambeta<double>(V)),Nah,GNACHAN,VNACHAN);


  const int t2 = TIME;
  //const int t1 = t2 % 400;

  //if (t2>10000 && t1>200) dV += injCur;
  //if (t2>10000) dV += injCur;


  dV += (*drivinp);
  dV /=CAPAC;

  dNasm = ratefcn<double>(Nasm,sigmoid<float>(V,NASMTHE,NASMK),
			  TadjAdj<double>(taufcn<float>(V,NASMTAU1,NASMPHI,NASMSIG0),TADJPNAS));

  dNapm = ratefcn<double>(Napm,sigmoid<float>(V,NAPMTHE,NAPMK),TadjAdj<double>(NAPMTAU1,TADJPKAF));

  dKrpm = ratefcn<double>(Krpm,sigmoid<float>(V,KRPMTHE,KRPMK),
			  TadjAdj<double>(taufcn<float>(V,KRPMTAU1,KRPMPHI,KRPMSIG0),TADJPKAF));

  const double val1 = Ashtaufcn(V);

  dKrph = ratefcn<double>(Krph,sigmoid<float>(V,KRPHTHE,KRPHK),TadjAdj<double>(val1*3.0,TADJPKAF));

  dAsh = ratefcn<double>(Ash,sigmoid<float>(V,ASHTHE,ASHK),TadjAdj<double>(val1,TADJPKAF));

  dAsm = ratefcn<double>(Asm,sigmoid<float>(V,XASMTHE,XASMK),
			 TadjAdj<double>(taufcn<float>(V,XASMTAU1,XASMPHI,XASMSIG0),TADJPKAF));

  dAfm = ratefcn<double>(Afm,sigmoid<float>(V,AFMTHE,AFMK),TadjAdj<double>(AFMTAU1,TADJPKAF));

  dAfh = ratefcn<double>(Afh,sigmoid<float>(V,AFHTHE,AFHK),TadjAdj<double>(AFHTAU1,TADJPKAF));

  //dKirm = ratefcn<double>(Kirm,sigmoid<float>(V,KIRMTHE,KIRMK),TadjAdj<double>(KIRMTAU1,TADJPKAF));

  dKm = gatefcn<double>(Km,Kmalpha<double>(V),Kmbeta<double>(V),5.0);

  dNah = gatefcn<double>(Nah,Nahalpha<double>(V),Nahbeta<double>(V),5.0);


}
 

void MahonUnit::initialize(RNG& rng) 
{

  //s1.x.increaseSizeTo(NUMVARS);
  //nodeVars.increaseSizeTo(NUMVARS);  
  initializeIterator(NUMVARS,SHD.deltaT);
  
  //ShallowArray<double> & x = s1.x;//nodeVars;

  ShallowArray<double> & x = Vars();

  double & V = x[0];
  double & Nasm = x[1];
  double & Napm = x[2];
  double & Krpm = x[3];
  double & Krph = x[4];
  double & Asm = x[5];
  double & Ash = x[6];
  double & Afm = x[7];
  double & Afh = x[8];
  
  double & Km = x[9];  
  double & Nah = x[10];
  //double & Kirm = x[11];

  V=V_init;
  Nasm = sigmoid<float>(V,NASMTHE,NASMK);
  Napm = sigmoid<float>(V,NAPMTHE,NAPMK);
  Krpm = sigmoid<float>(V,KRPMTHE,KRPMK);
  Krph = sigmoid<float>(V,KRPHTHE,KRPHK);
  Ash = sigmoid<float>(V,ASHTHE,ASHK);
  Asm = sigmoid<float>(V,XASMTHE,XASMK);
  Afm = sigmoid<float>(V,AFMTHE,AFMK);
  Afh = sigmoid<float>(V,AFHTHE,AFHK);
  //Kirm = sigmoid<float>(V,KIRMTHE,KIRMK);
  Km = gatefcnInstant<double>(Kmalpha<double>(V),Kmbeta<double>(V));
  Nah = gatefcnInstant<double>(Nahalpha<double>(V),Nahbeta<double>(V));

  //nodeVars[12] = g_init;
  //nodeVars[2] = n_init;
    
  std::ofstream ofs;
  ofs.open ("weights.dat", std::ofstream::out | std::ofstream::app);
  outputWeights(ofs);
  ofs.close();
  
  /*
  std::ofstream ofs;
  ofs.open ("inps.dat", std::ofstream::out | std::ofstream::app);
  outputDrivInp(ofs);
  ofs.close();
  */

}


void MahonUnit::update1(RNG& rng) 
{
  callIteratePhase1(rk1);
  //callIterate(s1.x);
}

void MahonUnit::update2(RNG& rng) 
{
  callIteratePhase2(rk1);
  //callIterate(s1.x);
}

void MahonUnit::update3(RNG& rng) 
{
  callIteratePhase3(rk1);
  //callIterate(s1.x);
}

void MahonUnit::update4(RNG& rng) 
{
  callIteratePhase4(rk1);
  //callIterate(s1.x);
}


/*
void MahonUnit::update(RNG& rng) 
{
  callIterate(); 
}
*/

void MahonUnit::flushVars(const ShallowArray< double > & x)
{
  g_out = x[11];
}

//void MahonUnit::flushVars_x1() 
//{
//  g_out = x1[11];
//}



void MahonUnit::flushVars1(RNG& rng) 
{

  prePhase1(rk1);

}

void MahonUnit::flushVars2(RNG& rng) 
{
  prePhase2(rk1);

}

void MahonUnit::flushVars3(RNG& rng) 
{
  prePhase3(rk1);

}

void MahonUnit::flushVars4(RNG& rng) 
{
  prePhase4(rk1);
 
}



void MahonUnit::updateOutputs(RNG& rng) 
{

  ShallowArray<double> & x = Vars();
  if (x[0]>= SHD.spikethresh && var1 < SHD.spikethresh) spike = true;
  else spike = false;
  var1 = x[0];
  var2 = x[5];
  var3 = x[11];
}




bool MahonUnit::ConnectP1(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_MahonUnitInAttrPSet* CG_inAttrPset, CG_MahonUnitOutAttrPSet* CG_outAttrPset) 
{ 
  //unsigned n=3967;//getGridLayerData()->getNbrUnits();

  /*
  unsigned preIdx=CG_node->getNode()->getGlobalIndex();
  unsigned postIdx=getNode()->getGlobalIndex();
  //double in;
  unsigned inWeightIdx=(postIdx*1000)+(preIdx);
  long connectionSeed=CG_inAttrPset->connectionSeed;
  for (int i=0;i<inWeightIdx;i++) ran0(&connectionSeed);
  float in = ran0(&connectionSeed);
  */

   RNG rng; // use a temporary local version only here: TODO update, not efficient on the GPU.
      // (seed for the in->out connection)
      //rng.reSeedShared(getSimulation().getRandomSeed()
 //               + (getNode()->getIndex()
 //                       * getGridLayerData()->getNbrUnits())
 //                    + CG_node->getNode()->getIndex());


  
  unsigned preIdx=CG_node->getNode()->getIndex();
  unsigned postIdx=getNode()->getIndex();
  //double in;
  unsigned inWeightIdx=(postIdx*89423217)+(preIdx*37);
  rng.reSeedShared(connectionSeed+inWeightIdx);
  float in = drandom(rng);    
  



  //std::cout << postIdx << " " << preIdx << " ";
  //RNG rng; // use a temporary local version only here
  //unsigned connectionSeed=CG_inAttrPset->connectionSeed;
  //rng.reSeedShared(connectionSeed+inWeightIdx);
  //in=drandom(rng);
  if (in<CG_inAttrPset->connectionProb) return true;
  return false;
}


MahonUnit::~MahonUnit() 
{
}

/*

void MahonUnit::derivs(const ShallowArray< double > & x, ShallowArray< double > & dx)
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

*/

void MahonUnit::setIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_MahonUnitInAttrPSet* CG_inAttrPset, CG_MahonUnitOutAttrPSet* CG_outAttrPset) 
{

  MSNNetInps[MSNNetInps.size()-1].row = getGlobalIndex()+1; // +1 is for Matlab
  MSNNetInps[MSNNetInps.size()-1].col = CG_node->getGlobalIndex()+1; 

}





void MahonUnit::setLypIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_MahonUnitInAttrPSet* CG_inAttrPset, CG_MahonUnitOutAttrPSet* CG_outAttrPset) 
{
  std::ofstream ofs;
  ofs.open ("MahonLypcons.dat", std::ofstream::out | std::ofstream::app);
  ofs << getGlobalIndex() << " " << CG_node->getGlobalIndex() << std::endl;
  ofs.close();
}


void MahonUnit::outputWeights(std::ofstream& fs)
{
  ShallowArray<Input>::iterator iter, end=MSNNetInps.end();

  for (iter=MSNNetInps.begin(); iter!=end; ++iter)
    fs<<iter->row<<" "<<iter->col<<" "<<iter->weight<<std::endl;
}


/*
void MahonUnit::outputDrivInp(std::ofstream& fs)
{
  fs << (*drivinp) << std::endl;
}
*/
