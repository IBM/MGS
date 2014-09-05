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

#ifndef NODETYPE_H
#define NODETYPE_H
#include "Copyright.h"

#include <memory>
#include <string>


class ParameterSet;
class NodeAccessor;
class GridLayerDescriptor;
class ConnectionIncrement;

class NodeType
{

   public:
      virtual void getNodeAccessor(std::auto_ptr<NodeAccessor> & r_aptr, GridLayerDescriptor* gridLayerDescriptor) =0;
      virtual void getInitializationParameterSet(std::auto_ptr<ParameterSet> & r_aptr) =0;
      virtual void getInAttrParameterSet(std::auto_ptr<ParameterSet> & r_aptr) =0;
      virtual void getOutAttrParameterSet(std::auto_ptr<ParameterSet> & r_aptr) =0;
      virtual std::string getModelName() =0;
      virtual ConnectionIncrement* getComputeCost() = 0;
      virtual ~NodeType() {}
};
#endif
