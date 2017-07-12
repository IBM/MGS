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

#include "C_returnType.h"
#include "MdlContext.h"
#include "C_typeClassifier.h"
#include "VoidType.h"
#include "DataType.h"
#include <memory>

void C_returnType::execute(MdlContext* context) 
{
   if (isVoid()) {
      delete _dataType;
      _dataType = new VoidType();
   } else {
      _type->execute(context);
      std::auto_ptr<DataType> dt;
      _type->releaseDataType(dt);
      delete _dataType;
      _dataType = dt.release();      
   }
}

C_returnType::C_returnType(bool v)
   : C_production(), _void(v), _type(0), _dataType(0)
{

}

C_returnType::C_returnType(C_typeClassifier* t)
   : C_production(), _void(false), _type(t), _dataType(0)
{
}

C_returnType::C_returnType(const C_returnType& rv) 
   : C_production(), _void(rv._void), _type(0), _dataType(0)
{
   if (rv._type) {
      std::auto_ptr<C_typeClassifier> dup;
      rv._type->duplicate(dup);
      _type = dup.release();
   }
   if (rv._dataType) {
      std::auto_ptr<DataType> dup;
      rv._dataType->duplicate(dup);
      _dataType = dup.release();
   }
}

void C_returnType::duplicate(std::auto_ptr<C_returnType>& rv) const
{
   rv.reset(new C_returnType(*this));
}

bool C_returnType::isVoid() const
{
   return _void;
}

C_typeClassifier* C_returnType::getType() const
{
   return _type;
}

void C_returnType::releaseDataType(std::auto_ptr<DataType>& dt) 
{
   dt.reset(_dataType);
   _dataType = 0;
}

C_returnType::~C_returnType() 
{
   delete _type;
   delete _dataType;
}


