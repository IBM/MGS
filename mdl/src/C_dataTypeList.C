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
      std::auto_ptr<std::vector<DataType*> > dtv;
      _dataTypeList->releaseDataTypeVec(dtv);
      _dataTypeVec = dtv.release();      
   } else {
      _dataTypeVec = new std::vector<DataType*>();
   }
   std::auto_ptr<std::vector<DataType*> > dtVec;
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
      std::auto_ptr<C_dataType> dup;
      rv._dataType->duplicate(dup);
      _dataType = dup.release();
   }
   if (rv._dataTypeList) {
      std::auto_ptr<C_dataTypeList> dup;
      rv._dataTypeList->duplicate(dup);
      _dataTypeList = dup.release();
   }
   deepCopyVector(rv);
}

void C_dataTypeList::duplicate(std::auto_ptr<C_dataTypeList>& rv) const
{
   rv.reset(new C_dataTypeList(*this));
}

void C_dataTypeList::releaseDataTypeVec(
   std::auto_ptr<std::vector<DataType*> >& dtv) 
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
      std::auto_ptr<DataType> dup;   
      for (it = rv._dataTypeVec->begin(); it != end; it++) {
	 (*it)->duplicate(dup);
	 _dataTypeVec->push_back(dup.release());
      }
   }
}
