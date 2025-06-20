// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_dataTypeList.h"
#include "MdlContext.h"
#include "C_dataType.h"
#include "DataType.h"
#include "InternalException.h"
#include <memory>
#include <vector>

void C_dataTypeList::execute(MdlContext* context) 
{
   if (_dataType == 0) {
      throw InternalException("_dataType is 0 in C_dataTypeList::execute");
   }
   _dataType->execute(context);
   if (_dataTypeList) {
      _dataTypeList->execute(context);
      deleteVector();
      std::unique_ptr<std::vector<DataType*> > dtv;
      _dataTypeList->releaseDataTypeVec(dtv);
      _dataTypeVec = dtv.release();      
   } else {
      _dataTypeVec = new std::vector<DataType*>();
   }
   std::unique_ptr<std::vector<DataType*> > dtVec;
   _dataType->releaseDataTypeVec(dtVec);
   _dataTypeVec->insert(_dataTypeVec->end(), dtVec->begin(), dtVec->end());
}

C_dataTypeList::C_dataTypeList() 
   : C_production(), _dataType(0), _dataTypeList(0), _dataTypeVec(0) 
{

}

C_dataTypeList::C_dataTypeList(C_dataType* dt) 
   : C_production(), _dataType(dt), _dataTypeList(0), _dataTypeVec(0) 
{

}

C_dataTypeList::C_dataTypeList(C_dataTypeList* dtl, C_dataType* dt) 
   : C_production(), _dataType(dt), _dataTypeList(dtl), _dataTypeVec(0) 
{

}

C_dataTypeList::C_dataTypeList(const C_dataTypeList& rv) 
   : C_production(rv), _dataType(0), _dataTypeList(0), _dataTypeVec(0) 
{
   if (rv._dataType) {
      std::unique_ptr<C_dataType> dup;
      rv._dataType->duplicate(std::move(dup));
      _dataType = dup.release();
   }
   if (rv._dataTypeList) {
      std::unique_ptr<C_dataTypeList> dup;
      rv._dataTypeList->duplicate(std::move(dup));
      _dataTypeList = dup.release();
   }
   deepCopyVector(rv);
}

void C_dataTypeList::duplicate(std::unique_ptr<C_dataTypeList>&& rv) const
{
   rv.reset(new C_dataTypeList(*this));
}

void C_dataTypeList::releaseDataTypeVec(
   std::unique_ptr<std::vector<DataType*> >& dtv) 
{
   dtv.reset(_dataTypeVec);
   _dataTypeVec = 0;
}

C_dataTypeList::~C_dataTypeList() 
{
   delete _dataType;
   delete _dataTypeList;
   deleteVector();
}

void C_dataTypeList::deleteVector() 
{
   if (_dataTypeVec) {
      std::vector<DataType*>::iterator end = _dataTypeVec->end();
      std::vector<DataType*>::iterator it;
      for (it = _dataTypeVec->begin(); it != end; it++) {
	 delete *it;
      }
      delete _dataTypeVec;
      _dataTypeVec = 0;
   }
}

void C_dataTypeList::deepCopyVector(const C_dataTypeList& rv) 
{
   if (rv._dataTypeVec) {
      _dataTypeVec = new std::vector<DataType*>();
      std::vector<DataType*>::const_iterator end = rv._dataTypeVec->end();
      std::vector<DataType*>::const_iterator it;
      std::unique_ptr<DataType> dup;   
      for (it = rv._dataTypeVec->begin(); it != end; it++) {
	 (*it)->duplicate(std::move(dup));
	 _dataTypeVec->push_back(dup.release());
      }
   }
}
