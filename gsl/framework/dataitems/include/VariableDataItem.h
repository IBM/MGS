// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef VARIABLEDATAITEM_H
#define VARIABLEDATAITEM_H
#include "Copyright.h"

#include "TriggerableDataItem.h"
#include <vector>
#include <memory>

class Variable;
class VariableInstanceAccessor;
class Triggerable;

class VariableDataItem : public TriggerableDataItem
{

   private:
      VariableInstanceAccessor *_data;

   public:
      static char const* _type;

      VariableDataItem();
      VariableDataItem(Variable* data);

      ~VariableDataItem();

      void duplicate(std::unique_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      VariableInstanceAccessor* getVariable(Error* error=0) const;
      void setVariable(VariableInstanceAccessor* v, Error* error=0);
      std::string getString(Error* error=0) const;

      virtual std::vector<Triggerable*> getTriggerables();
};
#endif
