// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef SIMULATIONDATAITEM_H
#define SIMULATIONDATAITEM_H
#include "Copyright.h"

#include "DataItem.h"

class Simulation;

class SimulationDataItem : public DataItem
{
   private:
      Simulation *_simulation;

   public:
      static const char* _type;

      virtual SimulationDataItem& operator=(const SimulationDataItem& DI);

      // Constructors
      SimulationDataItem(Simulation *simulation = 0);
      SimulationDataItem(const SimulationDataItem& DI);

      // Destructor
      ~SimulationDataItem();

      // Utility methods
      void duplicate(std::unique_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      Simulation* getSimulation() const;
      void setSimulation(Simulation* sim);
      std::string getString(Error* error=0) const;

};
#endif
