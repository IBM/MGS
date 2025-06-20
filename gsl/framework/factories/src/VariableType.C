// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "VariableType.h"
#include "VariableDataItem.h"
#include "Variable.h"
#include "VariableInstanceAccessor.h"
#include "DataItemQueriable.h"
#include "InstanceFactoryQueriable.h"
#include "NDPairList.h"
#include "VariableGranuleMapper.h"
#include "GslContext.h"
#include "Simulation.h"
#include "ConnectionIncrement.h"

VariableType::VariableType() : InstanceFactory()
{
  _instanceAtEachMPIProcess = false;
}

void VariableType::getInstance(std::unique_ptr<DataItem>& adi,
                               std::vector<DataItem*> const* args,
                               GslContext* c)
{
  VariableDataItem* vdi = new VariableDataItem;
  VariableInstanceAccessor* via = new VariableInstanceAccessor;
  Variable* av;
  unsigned variableIndex;
  if (c->sim->isGranuleMapperPass() || c->sim->isCostAggregationPass())
  {
    variableIndex = c->sim->incrementCurrentVariableId();
    ConnectionIncrement* computeCost = 0;
    av = allocateVariable();
    via->setVariableIndex(variableIndex);
    via->setVariable(av);
    via->setVariableType(getCompCategoryBase());
    via->setVariableIndex(variableIndex);
    av->setVariableDescriptor(via);
    av->initialize(c, *args);
    computeCost = new ConnectionIncrement();
    computeCost->_computationTime = 1.;
    computeCost->_memoryBytes = 4;
    computeCost->_communicationBytes = 4;
    if (c->sim->isGranuleMapperPass())
      addGranuleToSimulation(*(c->sim), computeCost, variableIndex);
  }
  else
  {
    variableIndex = c->sim->incrementCurrentVariableId();
    c->sim->incrementGranuleMapperCountOnceForVariable();
    GranuleMapper* vgm = c->sim->getVariableGranuleMapper();
    unsigned currentSpaceId = c->sim->getRank();
    // TUAN: the problem is it always return zero for
    // vgm->getGranule(variableIndex)->getPartitionId()
    // so only rank 0 MPI process get created
    // BUG TODO
    if (this->_instanceAtEachMPIProcess == true)
    {
      av = allocateVariable();
      via->setVariableIndex(variableIndex);
      via->setVariable(av);
      via->setVariableType(getCompCategoryBase());
      av->setVariableDescriptor(via);
      av->initialize(c, *args);
    }
    else
    {
      if (vgm->getGranule(variableIndex)->getPartitionId() == currentSpaceId)
      {
        av = allocateVariable();
        via->setVariableIndex(variableIndex);
        via->setVariable(av);
        via->setVariableType(getCompCategoryBase());
        av->setVariableDescriptor(via);
        av->initialize(c, *args);
      }
      else
      {
        av = 0;
        via->setVariableIndex(variableIndex);
        via->setVariable(av);
        via->setVariableType(getCompCategoryBase());
      }
    }
  }
  vdi->setVariable(via);
  adi.reset(vdi);
}

void VariableType::getInstance(std::unique_ptr<DataItem>& adi,
                               const NDPairList& ndplist, GslContext* c)
{
  VariableDataItem* vdi = new VariableDataItem;
  VariableInstanceAccessor* via = new VariableInstanceAccessor;
  Variable* av;
  unsigned variableIndex;
  if (c->sim->isGranuleMapperPass() || c->sim->isCostAggregationPass())
  {
    variableIndex = c->sim->incrementCurrentVariableId();
    ConnectionIncrement* computeCost = 0;
    av = allocateVariable();
    via->setVariableIndex(variableIndex);
    via->setVariable(av);
    via->setVariableType(getCompCategoryBase());
    via->setVariableIndex(variableIndex);
    av->setVariableDescriptor(via);
    av->initialize(ndplist);
    computeCost = new ConnectionIncrement();
    computeCost->_computationTime = 1;
    computeCost->_memoryBytes = 4;
    computeCost->_communicationBytes = 4;
    if (c->sim->isGranuleMapperPass())
      addGranuleToSimulation(*(c->sim), computeCost, variableIndex);
  }
  else
  {
    variableIndex = c->sim->incrementCurrentVariableId();
    c->sim->incrementGranuleMapperCountOnceForVariable();
    GranuleMapper* vgm = c->sim->getVariableGranuleMapper();
    unsigned currentSpaceId = c->sim->getRank();
    // TUAN: the problem is it always return zero for
    // vgm->getGranule(variableIndex)->getPartitionId()
    // so only VariableTypes in rank 0 MPI process get created
    // BUG TODO
    if (this->_instanceAtEachMPIProcess == true)
    {
      av = allocateVariable();
      via->setVariableIndex(variableIndex);
      via->setVariable(av);
      via->setVariableType(getCompCategoryBase());
      av->setVariableDescriptor(via);
      av->initialize(ndplist);
    }
    else
    {
      if (vgm->getGranule(variableIndex)->getPartitionId() == currentSpaceId)
      {
        av = allocateVariable();
        via->setVariableIndex(variableIndex);
        via->setVariable(av);
        via->setVariableType(getCompCategoryBase());
        av->setVariableDescriptor(via);
        av->initialize(ndplist);
      }
      else
      {
        av = 0;
        via->setVariableIndex(variableIndex);
        via->setVariable(av);
        via->setVariableType(getCompCategoryBase());
      }
    }
  }
  vdi->setVariable(via);
  adi.reset(vdi);
}

VariableType::~VariableType() {}

void VariableType::addGranuleToSimulation(Simulation& sim,
                                          ConnectionIncrement* computeCost,
                                          unsigned variableIndex) const
{
  if (!sim.hasVariableGranuleMapper())
  {
    std::unique_ptr<GranuleMapper> granuleMapper;
    GranuleMapper* gm = new VariableGranuleMapper(&sim);
    granuleMapper.reset(gm);
    unsigned index = sim.getGranuleMapperCount();
    gm->setIndex(index);
    sim.setVariableGranuleMapperIndex(index);
    sim.addGranuleMapper(granuleMapper);
    sim.incrementGranuleMapperCount();
  }
  sim.getVariableGranuleMapper()->addGranule(new Granule(), variableIndex);
}
