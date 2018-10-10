// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
