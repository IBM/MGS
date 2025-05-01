// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef VariableInstanceAccessor_H
#define VariableInstanceAccessor_H
#include "Copyright.h"

#include "Variable.h"
#include "ShallowArray.h"
#include "VariableDescriptor.h"

#include <deque>
#include <vector>
#include <set>
#include <cassert>

class Constant;
class Edge;
class Variable;
class Simulation;
class VariableDescriptor;
class VariableCompCategoryBase;
class Publisher;


class VariableInstanceAccessor : public VariableDescriptor
{

   public:
      VariableInstanceAccessor();
      virtual Variable* getVariable();
      virtual void setVariable(Variable*);
      virtual int getVariableIndex() const;
      virtual void setVariableIndex(int pos);

      virtual ~VariableInstanceAccessor();

      virtual Publisher* getPublisher() {
	assert(_variable);
	return _variable->getPublisher();
      }
      virtual const char* getServiceName(void* data) const {
	return _variable->getServiceName(data);
      }
      virtual const char* getServiceDescription(void* data) const {
	return _variable->getServiceDescription(data);
      }
      virtual void setVariableType(VariableCompCategoryBase* vcb) {
         _compCategory = vcb;
      }
      virtual VariableCompCategoryBase* getVariableType() {
         return _compCategory;
      }

   protected:
      Variable* _variable;
      VariableCompCategoryBase* _compCategory;
      int _variableIndex;
};

#endif
