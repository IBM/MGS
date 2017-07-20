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

#include "C_typeClassifier.h"
#include "MdlContext.h"
#include "C_typeCore.h"
#include "C_array.h"
#include "InternalException.h"
#include "SyntaxErrorException.h"
#include <memory>

void C_typeClassifier::execute(MdlContext* context) 
{
   if (_typeCore != 0) {
      std::auto_ptr<DataType> dt;
      _typeCore->execute(context);
      _typeCore->releaseDataType(dt);
      _dataType = dt.release();
   } else if (_array != 0) {
      std::auto_ptr<DataType> dt;
      _array->execute(context);
      _array->releaseDataType(dt);
      _dataType = dt.release();
   } else {
      throw InternalException
	 ("Both _typeCore and _array are 0 in C_typeClassifier::execute");
   }
   try {
      _dataType->setPointer(_pointer);
   } catch (SyntaxErrorException& e) {
      e.setCaught();
      e.setFileName(getFileName());
      e.setLineNumber(getLineNumber());
      throw;
   }
}

C_typeClassifier::C_typeClassifier() 
   : C_production(), _typeCore(0), _array(0), _pointer(false), _dataType(0) 
{

}

C_typeClassifier::C_typeClassifier(C_typeCore* tc, bool pointer) 
   : C_production(), _typeCore(tc), _array(0), _pointer(pointer), _dataType(0) 
{   
}

C_typeClassifier::C_typeClassifier(C_array* a, bool pointer) 
   : C_production(), _typeCore(0), _array(a), _pointer(pointer), _dataType(0) 
{

}

C_typeClassifier::C_typeClassifier(const C_typeClassifier& rv) 
   : C_production(rv), _typeCore(0), _array(0), _pointer(rv._pointer)
   , _dataType(0) 
{
   if (rv._typeCore) {
      std::auto_ptr<C_typeCore> dup;
      rv._typeCore->duplicate(dup);
      _typeCore = dup.release();
   }
   if (rv._array) {
      std::auto_ptr<C_array> dup;
      rv._array->duplicate(dup);
      _array = dup.release();
   }
   if (rv._dataType) {
      std::auto_ptr<DataType> dup;
      rv._dataType->duplicate(dup);
      _dataType = dup.release();
   }
}

void C_typeClassifier::duplicate(std::auto_ptr<C_typeClassifier>& rv) const
{
   rv.reset(new C_typeClassifier(*this));
}

void C_typeClassifier::releaseDataType(std::auto_ptr<DataType>& dt) 
{
   dt.reset(_dataType);
   _dataType = 0;
}

C_typeClassifier::~C_typeClassifier() 
{
   delete _typeCore;
   delete _array;
   delete _dataType;
}


