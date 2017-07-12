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

#include "C_typeCore.h"
#include "MdlContext.h"
#include "DataType.h"
#include "StructType.h"
#include "NotFoundException.h"
#include <memory>
#include <iostream>

void C_typeCore::execute(MdlContext* context) 
{
   if (_id != "") {
      try {
	 std::auto_ptr<DataType> dup;
	 StructType* st;
	 Generatable* gen = context->_generatables->getMember(_id);
	 st = dynamic_cast<StructType*>(gen);
	 if (st == 0) {
	    std::ostringstream stream;
	    stream << _id << " is found, but has a different type.";
	    throw NotFoundException(stream.str());	    
	 }
	 st->duplicate(dup);
	 delete _dataType;
	 _dataType = dup.release();
      } catch (NotFoundException& e) {
	 std::cerr << e.getError() << std::endl;
	 e.setError("");
	 throw;
      }
   }

}

C_typeCore::C_typeCore() 
   : C_production(), _dataType(0), _id("") 
{

}

C_typeCore::C_typeCore(DataType* dt) 
   : C_production(), _dataType(0), _id("")
{
   _dataType = dt;
}

C_typeCore::C_typeCore(const std::string& s) 
   : C_production(), _dataType(0), _id("") 
{
   _id = s;
}

C_typeCore::C_typeCore(const C_typeCore& rv) 
   : C_production(rv), _dataType(0), _id(rv._id) 
{
   if (rv._dataType) {
      std::auto_ptr<DataType> dup;
      rv._dataType->duplicate(dup);
      _dataType = dup.release();
   }
}

void C_typeCore::duplicate(std::auto_ptr<C_typeCore>& rv) const
{
   rv.reset(new C_typeCore(*this));
}

void C_typeCore::releaseDataType(std::auto_ptr<DataType>& dt) 
{
   dt.reset(_dataType);
   _dataType = 0;
}

C_typeCore::~C_typeCore() 
{
   delete _dataType;
}


