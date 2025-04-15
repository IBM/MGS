// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef SIMULATIONQUERIABLE_H
#define SIMULATIONQUERIABLE_H
//#include "Copyright.h"

#include "Queriable.h"

#include <list>
#include <memory>
#include <string>


class Simulation;
class QueryField;
class QueryResult;
class QueryDescriptor;

class SimulationQueriable : public Queriable
{

   public:
      SimulationQueriable(Simulation* simulation);
      SimulationQueriable(const SimulationQueriable&);
      std::unique_ptr<QueryResult> query(int maxtItem, int minItem, int searchSize);
      Publisher* getQPublisher();
      virtual void duplicate(std::unique_ptr<Queriable>& dup) const;
      void getDataItem(std::unique_ptr<DataItem> &);
      void refresh();
      ~SimulationQueriable();

   private:
      Simulation* _sim;
};
#endif
