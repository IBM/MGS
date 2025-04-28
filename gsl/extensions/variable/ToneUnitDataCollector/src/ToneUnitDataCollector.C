#include "Lens.h"
#include "ToneUnitDataCollector.h"
#include "CG_ToneUnitDataCollector.h"
#include <memory>

void ToneUnitDataCollector::initialize(RNG& rng) 
{
}

void ToneUnitDataCollector::finalize(RNG& rng) 
{
}

void ToneUnitDataCollector::dataCollection(Trigger* trigger, NDPairList* ndPairList) 
{
}

void ToneUnitDataCollector::getNodeIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ToneUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_ToneUnitDataCollectorOutAttrPSet* CG_outAttrPset) 
{
}

ToneUnitDataCollector::ToneUnitDataCollector() 
   : CG_ToneUnitDataCollector()
{
}

ToneUnitDataCollector::~ToneUnitDataCollector() 
{
}

void ToneUnitDataCollector::duplicate(std::unique_ptr<ToneUnitDataCollector>&& dup) const
{
   dup.reset(new ToneUnitDataCollector(*this));
}

void ToneUnitDataCollector::duplicate(std::unique_ptr<Variable>&& dup) const
{
   dup.reset(new ToneUnitDataCollector(*this));
}

void ToneUnitDataCollector::duplicate(std::unique_ptr<CG_ToneUnitDataCollector>&& dup) const
{
   dup.reset(new ToneUnitDataCollector(*this));
}

