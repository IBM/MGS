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

#ifndef Constant_H
#define Constant_H
//#include "Copyright.h"

#include <memory>
#include <vector>
#include "Publishable.h"

class DataItem;
class Edge;
class LensContext;
class NodeDescriptor;
class ParameterSet;
//class Variable;
class VariableDescriptor;
class NDPairList;

class Constant : public Publishable
{

   public:
      virtual void addPostEdge(Edge* e, ParameterSet* OutAttrPSet) = 0;
      virtual void addPostNode(NodeDescriptor* n, 
			       ParameterSet* OutAttrPSet) = 0;
      virtual void addPostVariable(VariableDescriptor* v, ParameterSet* OutAttrPSet) = 0;
      virtual ~Constant();
      virtual void getOutAttrParameterSet(
	 std::auto_ptr<ParameterSet>& outAttrPSet) const = 0;
      virtual void duplicate(std::auto_ptr<Constant>& dup) const = 0;
      void initialize(LensContext *c, const std::vector<DataItem*>& args);
      void initialize(const NDPairList& ndplist);
   protected:
      virtual void doInitialize(LensContext *c, 
				const std::vector<DataItem*>& args) = 0;
      virtual void doInitialize(const NDPairList& ndplist) = 0;
};

#endif
