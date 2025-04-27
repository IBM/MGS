#include "Lens.h"
#include "LypCollector.h"
#include "CG_LypCollector.h"
#include "rndm.h"
#include <fstream>


void LypCollector::initialize(RNG& rng) 
{
 output = 0;
}

void LypCollector::update(RNG& rng) 
{
  output = 0;
  ShallowArray<double *>::iterator iter, end=inputs.end();
  for (iter=inputs.begin(); iter!=end; ++iter) output+=**iter;
}

void LypCollector::setIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_LypCollectorInAttrPSet* CG_inAttrPset, CG_LypCollectorOutAttrPSet* CG_outAttrPset) 
{
  std::ofstream ofs;
  ofs.open ("GlobColCons.dat", std::ofstream::out | std::ofstream::app);
  ofs << getGlobalIndex() << " " << CG_node->getGlobalIndex() << std::endl;
  ofs.close();

}

LypCollector::~LypCollector() 
{
}

