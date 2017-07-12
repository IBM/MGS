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
   std::auto_ptr<DataType> dt;
   _typeClassifier->releaseDataType(dt);
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
      std::auto_ptr<C_typeClassifier> dup;
      rv._typeClassifier->duplicate(dup);
      _typeClassifier = dup.release();
   }
   if (rv._arrayType) {
      std::auto_ptr<DataType> dup;
      rv._arrayType->duplicate(dup);
      DataType* dt = dup.release();
      _arrayType = dynamic_cast<ArrayType*>(dt);
      if (_arrayType == 0) {
      throw InternalException (
	 "_arrayType is 0 in C_array::C_array(C_array* rv)");
      }
   }
}

void C_array::duplicate(std::auto_ptr<C_array>& rv) const
{
   rv.reset(new C_array(*this));
}

void C_array::releaseDataType(std::auto_ptr<DataType>& dt) 
{
   dt.reset(_arrayType);
   _arrayType = 0;
}


C_array::~C_array() 
{
   delete _typeClassifier;
   delete _arrayType;
}


