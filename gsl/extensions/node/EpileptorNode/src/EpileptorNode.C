#include "Lens.h"
#include "EpileptorNode.h"
#include "CG_EpileptorNode.h"
#include "rndm.h"
#include <cmath>
#include <map>
#include <pair>

#ifdef HAVE_GPU
#include "CG_EpileptorNodeCompCategory.h"
#endif

#define ITER getSimulation().getIteration()
#define SHD getSharedMembers()
#define GAMMA 0.01

#ifdef HAVE_GPU


#define x1  (_container->um_x1[index])
#define x2  (_container->um_x2[index])
#define y1  (_container->um_y1[index])
#define y2  (_container->um_y2[index])
#define z  (_container->um_z[index])
#define x0  (_container->um_x0[index])
#define u  (_container->um_u[index])
#define h  (_container->um_h[index])
#define dx1  (_container->um_dx1[index])
#define dx2  (_container->um_dx2[index])
#define dy1  (_container->um_dy1[index])
#define dy2  (_container->um_dy2[index])
#define dz  (_container->um_dz[index])
#define du  (_container->um_du[index])
#define K  (_container->um_K[index])
#endif



void EpileptorNode::initialize(RNG& rng) 
{
}

void EpileptorNode::updateDeltas(RNG& rng) 
{
  float f1 = (x1<0) ? ( (x1*x1*x1)-(3.0*x1*x1) ) : ( (x2-0.6*(z-4.0)*(z-4.0) ) * x1 );
  dx1 = y1 - f1 - z + SHD.I1;
  dy1 = SHD.y0 - 5.0*x1*x1 -y1;

#ifdef Proix_et_al_2014
  float h = SHD.x0 + 3.0/(1.0+exp((-x1-0.5)/0.1));
  float coupling = 0;
  auto end = K.end();
  for (auto iter=K.begin(); iter!=end; ++iter) {
    coupling += iter->weight * (*(iter->input) - x1);
  }
  dz = (1.0/SHD.tau_0) * (h-z) - coupling;
#elif defined Jirsa_et_al_2014
  dz = (1.0/SHD.tau_0) * (4.0 * (x1-x0) - z);
#endif
  
  du = -GAMMA * (u-0.1*x1); // See Jirsa et al. 2014, Supplementary material
                            // in which , 0.002g become 2u :
  dx2 = -y2 + x2 - x2*x2*x2 + SHD.I2 + 2.0*u - 0.3*(z-3.5); 
  float f2 = (x2<-0.25) ? 0 : (6.0 * (x2+0.25) );
  dy2 = (1.0/SHD.tau_2) * (-y2 + f2);
}

void EpileptorNode::update(RNG& rng) 
{
  x1 += dx1;
  x2 += dx2;
  y1 += dy1;
  y2 += dy2;
  z += dz;
  u += du;
}

void EpileptorNode::setWeight(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_EpileptorNodeInAttrPSet* CG_inAttrPset, CG_EpileptorNodeOutAttrPSet* CG_outAttrPset) 
{
  unsigned row = getGlobalIndex()+1; // +1 is for Matlab
  unsigned col = CG_node->getGlobalIndex()+1;
  K[K.size()-1].weight=connectionMap[std::make_pair(row, col)];
}

bool EpileptorNode::checkConnection(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_EpileptorNodeInAttrPSet* CG_inAttrPset, CG_EpileptorNodeOutAttrPSet* CG_outAttrPset) 
{
  bool rval = true;
  unsigned row = getGlobalIndex()+1; // +1 is for Matlab
  unsigned col = CG_node->getGlobalIndex()+1;
  if (connectionMap.size()==0) {
    auto end=CG_inAttrPset->connectionMatrix->end();
    for (auto iter=CG_inAttrPset->connectionMatrix->begin(); iter!=end; ++iter) {
      connectionMap[std::make_pair(int(*iter), int(*++iter))]=*++iter;
    }
  }
  auto mapIter = connectionMap.find(make_pair(row, col));
  if (mapIter==connectionMap.end()) {
    rval = false;
  }
  return rval;
}

EpileptorNode::~EpileptorNode() 
{
}

