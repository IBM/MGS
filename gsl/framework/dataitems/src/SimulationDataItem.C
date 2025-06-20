// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "SimulationDataItem.h"
#include "Simulation.h"

// Type
const char* SimulationDataItem::_type = "SIMULATION";

// Constructors
SimulationDataItem::SimulationDataItem(Simulation *simulation) 
   : _simulation(simulation)
{
}


SimulationDataItem::SimulationDataItem(const SimulationDataItem& DI)
{
   _simulation = DI._simulation;
}


// Utility methods
void SimulationDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(new SimulationDataItem(*this));
}


SimulationDataItem& SimulationDataItem::operator=(const SimulationDataItem& DI)
{
   _simulation = DI.getSimulation();
   return(*this);
}


const char* SimulationDataItem::getType() const
{
   return _type;
}


// Singlet methods

Simulation* SimulationDataItem::getSimulation() const
{
   return _simulation;
}


void SimulationDataItem::setSimulation(Simulation* sim)
{
   _simulation = sim;
}


SimulationDataItem::~SimulationDataItem()
{
}


std::string SimulationDataItem::getString(Error* error) const
{
   return "";
}
