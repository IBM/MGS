// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_typeClassifier_H
#define C_typeClassifier_H
#include "Mdl.h"

#include "C_production.h"
#include <memory>

class MdlContext;
class C_typeCore;
class C_array;
class DataType;

class C_typeClassifier : public C_production {
   using C_production::duplicate;  // Make base class method visible
   public:
      virtual void execute(MdlContext* context);
      C_typeClassifier();
      C_typeClassifier(C_typeCore* tc, bool pointer = false);
      C_typeClassifier(C_array* a, bool pointer = false);
      C_typeClassifier(const C_typeClassifier& rv);
      virtual void duplicate(std::unique_ptr<C_typeClassifier>&& rv) const;
      void releaseDataType(std::unique_ptr<DataType>&& dt);
      virtual ~C_typeClassifier();

   private:
      C_typeCore* _typeCore;
      C_array* _array;
      bool _pointer;
      DataType* _dataType;
};


#endif // C_typeClassifier_H
