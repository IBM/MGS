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

#ifndef TYPECLASSIFIER_H
#define TYPECLASSIFIER_H
#include "Copyright.h"

#include <string>
#include <vector>

class ConstantType;
class NodeType;
class EdgeType;
class FunctorType;
class TriggerType;
class StructType;
class VariableType;
class NodeSet;

template <class T> class TypeClassifier {
   public:
      inline static const std::string getName() {
	 return "N/A(Check TypeClassifier)";
      }
};

#endif
