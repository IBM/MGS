// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_returnType_H
#define C_returnType_H
#include "Mdl.h"

#include "C_production.h"
#include <memory>

class MdlContext;
class C_typeClassifier;
class DataType;

class C_returnType : public C_production {
   using C_production::duplicate;  // Make base class method visible
   public:
      virtual void execute(MdlContext* context);
      C_returnType(bool v = false);
      C_returnType(C_typeClassifier* _t);
      C_returnType(const C_returnType& rv);
      virtual void duplicate(std::unique_ptr<C_returnType>&& rv) const;
      void releaseDataType(std::unique_ptr<DataType>&& dt);
      virtual ~C_returnType();

      bool isVoid() const;
      C_typeClassifier* getType() const;

   private:
      bool _void;
      C_typeClassifier* _type;
      DataType* _dataType;
};


#endif // C_returnType_H
