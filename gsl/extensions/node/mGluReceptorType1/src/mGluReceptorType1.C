#include "Lens.h"
#include "mGluReceptorType1.h"
#include "CG_mGluReceptorType1.h"
#include "rndm.h"

#define ALPHA (getSharedMembers().alpha)
#define BETA (getSharedMembers().beta)

void mGluReceptorType1::initialize(RNG& rng) {}

void mGluReceptorType1::update(RNG& rng)
{
  // implement IP3 production = function of (*Glut)
  //dyn_var_t p = 1.0;    // [pA/(um^2 * uM)]
  dyn_var_t p = sigmoid(ALPHA, BETA);    // [pA/(um^2 * uM)]
  I_IP3 = (*Glut) * p;  // [pA/um^2]
}

void mGluReceptorType1::setPostIndex(
    const CustomString& CG_direction, const CustomString& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant, CG_mGluReceptorType1InAttrPSet* CG_inAttrPset,
    CG_mGluReceptorType1OutAttrPSet* CG_outAttrPset)
{
  indexPost = CG_inAttrPset->idx;
  if (indexPrePost.size() % 2)
  {//it means that PreSynapticPoint is being used
#ifdef KEEP_PAIR_PRE_POST
    indexPrePost.push_back(&indexPost);
#endif
  }
}

mGluReceptorType1::~mGluReceptorType1() {}

dyn_var_t mGluReceptorType1::sigmoid(dyn_var_t alpha, dyn_var_t beta)
{
  return exp(beta * alpha) / (1 + exp(beta * alpha));
}

void mGluReceptorType1::setPrePostIndex(
    const CustomString& CG_direction, const CustomString& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant, CG_mGluReceptorType1InAttrPSet* CG_inAttrPset,
    CG_mGluReceptorType1OutAttrPSet* CG_outAttrPset)
{
  indexPrePost.push_back(&(*(getSharedMembers().indexPrePost_connect))[0]);
  indexPrePost.push_back(&(*(getSharedMembers().indexPrePost_connect))[1]);
}

#ifdef KEEP_PAIR_PRE_POST
//do nothing
#else
void mGluReceptorType1::setPreIndex(
    const CustomString& CG_direction, const CustomString& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant, CG_mGluReceptorType1InAttrPSet* CG_inAttrPset,
    CG_mGluReceptorType1OutAttrPSet* CG_outAttrPset)
{
  indexPre = *(getSharedMembers().IntTmpConnect);
}
#endif
