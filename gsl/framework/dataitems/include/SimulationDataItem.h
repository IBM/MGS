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
      void duplicate(std::auto_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      Simulation* getSimulation() const;
      void setSimulation(Simulation* sim);
      std::string getString(Error* error=0) const;

};
#endif
