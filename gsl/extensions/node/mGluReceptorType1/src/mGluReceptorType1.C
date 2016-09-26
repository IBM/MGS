#include "Lens.h"
#include "mGluReceptorType1.h"
#include "CG_mGluReceptorType1.h"
#include "rndm.h"

void mGluReceptorType1::initialize(RNG& rng) {}

void mGluReceptorType1::update(RNG& rng)
{
  // implement IP3 production = function of (*Glut)
  //dyn_var_t p = 1.0;    // [pA/(um^2 * uM)]
  dyn_var_t p = sigmoid(alpha, beta);    // [pA/(um^2 * uM)]
  I_IP3 = (*Glut) * p;  // [pA/um^2]
}

void mGluReceptorType1::setPostIndex(
    const String& CG_direction, const String& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant, CG_mGluReceptorType1InAttrPSet* CG_inAttrPset,
    CG_mGluReceptorType1OutAttrPSet* CG_outAttrPset)
{
  indexPost = CG_inAttrPset->idx;
  indexPrePost.push_back(&indexPost);
}

mGluReceptorType1::~mGluReceptorType1() {}

dyn_var_t mGluReceptorType1::sigmoid(dyn_var_t alpha, dyn_var_t beta)
{
  return exp(beta * alpha) / (1 + exp(beta * alpha));
}
