// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef EdgeCompCategoryBase_H
#define EdgeCompCategoryBase_H
#include "Copyright.h"

#include <memory>
#include <string>
#include <map>
#include <deque>

#include "CompCategoryBase.h"
#include "EdgeType.h"

class ParameterSet;
class Simulation;
class WorkUnit;
class EdgePartitionItem;

class EdgeCompCategoryBase : public CompCategoryBase, public EdgeType
{

   public:
      EdgeCompCategoryBase(Simulation& sim, const std::string& modelName);
      virtual ~EdgeCompCategoryBase();

      // EdgeType functions to be implemented
      virtual void getInitializationParameterSet(
	 std::auto_ptr<ParameterSet>& initPSet) = 0;
      virtual void getInAttrParameterSet(
	 std::auto_ptr<ParameterSet>& inAttrPSet) = 0;
      virtual void getOutAttrParameterSet(
	 std::auto_ptr<ParameterSet>& outAttrPSet) = 0;
      virtual std::string getModelName() {
	 return _modelName;
      }

      // virtual void store(std::ostream&) =0;
      // virtual void reload(std::istream&) =0;

      virtual int initPartitions(int num);

   protected:

      EdgePartitionItem* _partitions;
      int _nbrPartitions;

      std::string _modelName;
      // ShallowArray<Edge*> _edgeList;
      virtual int getNumOfEdges() = 0;
      
   private:
//       // Disable 
//       EdgeCompCategoryBase(const EdgeCompCategoryBase& rv)
// 	 : CompCategoryBase(rv), EdgeType(rv) {}
//       // Disable
//       EdgeCompCategoryBase& operator=(
// 	 const EdgeCompCategoryBase& rv) {
// 	 return *this;
//       }    
};
#endif
