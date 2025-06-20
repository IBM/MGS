// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
