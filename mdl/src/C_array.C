// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_array.h"
#include "MdlContext.h"
#include "DataType.h"
#include "ArrayType.h"
#include "C_typeClassifier.h"
#include "InternalException.h"
#include "SyntaxErrorException.h"
#include <memory>

void C_array::execute(MdlContext* context) 
{
   
   if (_typeClassifier == 0) {
      throw InternalException ("_typeClassifier is 0 in C_array::execute");
   }
   _typeClassifier->execute(context);
   std::unique_ptr<DataType> dt;
   _typeClassifier->releaseDataType(std::move(dt));
   delete _arrayType;
   _arrayType = new ArrayType(dt.release());
}

C_array::C_array() 
   : C_production(), _typeClassifier(0), _arrayType(0)
{

}

C_array::C_array(C_typeClassifier* tc) 
   : C_production(), _typeClassifier(tc), _arrayType(0)
{

}

C_array::C_array(const C_array& rv) 
   : C_production(rv), _typeClassifier(0), _arrayType(0)
{
   if (rv._typeClassifier) {
      std::unique_ptr<C_typeClassifier> dup;
      rv._typeClassifier->duplicate(std::move(dup));
      _typeClassifier = dup.release();
   }
   if (rv._arrayType) {
      std::unique_ptr<DataType> dup;
      rv._arrayType->duplicate(std::move(dup));
      DataType* dt = dup.release();
      _arrayType = dynamic_cast<ArrayType*>(dt);
      if (_arrayType == 0) {
      throw InternalException (
	 "_arrayType is 0 in C_array::C_array(C_array* rv)");
      }
   }
}

void C_array::duplicate(std::unique_ptr<C_array>&& rv) const
{
   rv.reset(new C_array(*this));
}

void C_array::releaseDataType(std::unique_ptr<DataType>&& dt) 
{
   dt.reset(_arrayType);
   _arrayType = 0;
}


C_array::~C_array() 
{
   delete _typeClassifier;
   delete _arrayType;
}


