// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_array_H
#define C_array_H
#include "Mdl.h"

#include "C_production.h"
#include <memory>

class MdlContext;
class C_typeClassifier;
class DataType;
class ArrayType;

class C_array : public C_production {
   protected:
      using C_production::duplicate;  // Make base class method visible
   public:
      virtual void execute(MdlContext* context);
      C_array();
      C_array(C_typeClassifier* tc);
      C_array(const C_array& rv);
      virtual void duplicate(std::unique_ptr<C_array>&& rv) const;
      void releaseDataType(std::unique_ptr<DataType>&& dt);
      virtual ~C_array();

   private:
      C_typeClassifier* _typeClassifier;
      ArrayType* _arrayType;
};


#endif // C_array_H
