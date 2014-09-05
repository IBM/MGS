#ifndef VariableDescriptor_H
#define VariableDescriptor_H
#include "Copyright.h"

#include <vector>
#include "ShallowArray.h"
#include "Publishable.h"

class Variable;
class VariableCompCategoryBase;

class VariableDescriptor : public Publishable
{
   public:
      virtual Variable* getVariable() = 0;
      virtual void setVariable(Variable*) = 0;
      virtual int getVariableIndex() const = 0;
      virtual void setVariableIndex(int pos) = 0;
      virtual void setVariableType(VariableCompCategoryBase*) = 0;
      virtual VariableCompCategoryBase* getVariableType() = 0;
};

#endif
