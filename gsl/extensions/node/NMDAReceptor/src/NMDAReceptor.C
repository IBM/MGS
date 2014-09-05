// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "NMDAReceptor.h"
#include "CG_NMDAReceptor.h"
#include "rndm.h"
#include <iostream>
#include "math.h"
#include <limits>

#define ALPHA (getSharedMembers().alpha)
#define BETA (getSharedMembers().beta)
#define NEUROTRANSMITTER (getSharedMembers().Tmax/(1.0 + exp(-(*Vpre - getSharedMembers().Vp)/getSharedMembers().Kp)))
#define MGBLOCK (1.0/(1.0 + exp(-0.062*((*Vpost)[indexPost]))*(*(getSharedMembers().Mg_EC))/3.57))
//#define MGBLOCK 1.0/(1.0 + exp(-0.122*((*Vpost)[indexPost]))*(*(getSharedMembers().Mg_EC))/3.57) //Adjusted sigmoid to not get calcium transients at -60mV
#define DT (*(getSharedMembers().deltaT))
#define KETAMINE (*(getSharedMembers().Ketamine))

#define W w
#define pOn (getSharedMembers().plasticityOn)
#define pStart (getSharedMembers().plasticityStartAt)
#define pStop (getSharedMembers().plasticityStopAt)
#define TAU (100.0/(100.0/0.001+pow((*Ca_IC)[indexPost],3)) + 1000.0)
#define CAFUN (0.25+sigmoid((*Ca_IC)[indexPost]-0.55,80) - 0.25*sigmoid((*Ca_IC)[indexPost]-0.35,80))

void NMDAReceptor::initializeNMDA(RNG& rng) 
{
  assert(Vpre);
  assert(Vpost);
  assert(Ca_IC);
  assert(getSharedMembers().T!=0 && getSharedMembers().Ca_EC!=0 && getSharedMembers().Mg_EC!=0);

  if(KETAMINE==0){
    KETAMINE = 0;
  }

  float ALPHANEUROTRANSMITTER = ALPHA*NEUROTRANSMITTER;
  r = ALPHANEUROTRANSMITTER/(BETA + ALPHANEUROTRANSMITTER);
  g = w*gbar*MGBLOCK*r*(1-KETAMINE);

  buffer = 0;
  gbar0 = gbar;

 if(pOn){
   if(pOn==1){ //Graupner & Brunel 2012 PNAS
     tp = getSharedMembers().theta_p;
     }
    else if(pOn==2){ 
      tp = 0.55;
    }
  }
 }


void NMDAReceptor::updateNMDA(RNG& rng) 
{
  //Calculate receptor conductance
  float ALPHANEUROTRANSMITTER = ALPHA*NEUROTRANSMITTER;
  float A = DT*(BETA + ALPHANEUROTRANSMITTER)/2.0;
  r =  (DT*ALPHANEUROTRANSMITTER + r*(1.0 - A))/(1.0 + A);

  g = gbar*MGBLOCK*r*(1-KETAMINE);

  //Updates the channel reversal potential
  E_Ca = (0.04343 * *(getSharedMembers().T) * log(*(getSharedMembers().Ca_EC) / (*Ca_IC)[indexPost]));

  float gCa = g;
  if(pOn==1){
    gCa=g/10;}
  else if(pOn==2){
    gCa=g/20;}

  I_Ca = gCa*((*Vpost)[indexPost]-E_Ca);
}

void NMDAReceptor::updateNMDADepPlasticity(RNG& rng)
{
  if(pOn){
    if((getSimulation().getIteration()*DT)>pStart && (getSimulation().getIteration()*DT)<pStop){
      if(pOn==1){//Graupner & Brunel 2012 PNAS
	float dw = (-w*(1.0-w)*(getSharedMembers().w_th-w) + getSharedMembers().gamma_p*(1.0-w)*((float)(((*Ca_IC)[indexPost]-getSharedMembers().theta_p)>=0))  - getSharedMembers().gamma_d*w*((float)(((*Ca_IC)[indexPost]-getSharedMembers().theta_d)>=0)))/getSharedMembers().tau;
	w = w + DT*dw;

	if(getSharedMembers().deltaNMDAR){ //Metaplasticity
	  float dBuffer;

	  if(dw>0){dBuffer = -buffer + dw;}
	  else{dBuffer = -buffer;}

	  buffer = buffer + dBuffer*DT;
	  
	  float dgbar = (gbar0-gbar)/getSharedMembers().tauBuffer + getSharedMembers().alphaBuffer*buffer;
	  gbar = gbar + dgbar*DT;

	}

      }else if(pOn==2){ //Shouval & Bear & Cooper 2002 PNAS
	w = w + (1.0/TAU)*(CAFUN-w);
      }
     }
  }
}

void NMDAReceptor::setPostIndex(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_NMDAReceptorInAttrPSet* CG_inAttrPset, CG_NMDAReceptorOutAttrPSet* CG_outAttrPset) 
{
  indexPost = CG_inAttrPset->idx;
  indexPrePost.push_back(&indexPost);
}

float NMDAReceptor::sigmoid(float alpha, float beta){
  return exp(beta*alpha)/(1+exp(beta*alpha));
}

NMDAReceptor::~NMDAReceptor() 
{
}

