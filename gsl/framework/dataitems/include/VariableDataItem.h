// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
