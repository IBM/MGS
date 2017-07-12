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

#include "FunctorTypeDataItem.h"
#include "FunctorType.h"
#include "InstanceFactory.h"
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>

const char* FunctorTypeDataItem::_type = "FUNCTOR_TYPE";

FunctorTypeDataItem::FunctorTypeDataItem()
: _functorType(0), constructor_ptl(0), function_ptl(0), return_ptl(0), category("")
{
}


void FunctorTypeDataItem::setInstanceFactory(InstanceFactory* ifp )
{
   FunctorType *ftp = dynamic_cast<FunctorType*>(ifp);
   if(ftp ==0) {
      std::cerr<< "FunctorTypeDataItem:Unable to cast InstanceFactory to FunctorType!"<<std::endl;
      exit(-1);
   }
   setFunctorType(ftp);
}


InstanceFactory* FunctorTypeDataItem::getInstanceFactory() const
{
   return getFunctorType();
}


FunctorTypeDataItem::FunctorTypeDataItem(FunctorTypeDataItem const *f)
: InstanceFactoryDataItem(*f), _functorType(f->_functorType), constructor_ptl(f->constructor_ptl),
function_ptl(f->function_ptl), return_ptl(f->return_ptl), category(f->category)
{
}


FunctorTypeDataItem::~FunctorTypeDataItem()
{
}


void FunctorTypeDataItem::duplicate(std::auto_ptr<DataItem> & r_aptr) const
{
   DataItem *p= new FunctorTypeDataItem(this);
   r_aptr.reset(p);
}


const char* FunctorTypeDataItem::getType() const
{
   return _type;
}


FunctorType* FunctorTypeDataItem::getFunctorType() const
{
   return _functorType;
}


void FunctorTypeDataItem::setFunctorType(FunctorType* ft)
{
   _functorType = ft;
}


void FunctorTypeDataItem::setCategory(std::string cat)
{
   category = cat;
}


std::string FunctorTypeDataItem::getCategory() const
{
   return category;
}


void FunctorTypeDataItem::setConstructorParams(std::list<C_parameter_type> *ptl)
{
   constructor_ptl = ptl;
}


std::list<C_parameter_type> * FunctorTypeDataItem::getConstructorParams() const
{
   return constructor_ptl;
}


void FunctorTypeDataItem::setFunctionParams(std::list<C_parameter_type> *ptl)
{
   function_ptl = ptl;
}


std::list<C_parameter_type> * FunctorTypeDataItem::getFunctionParams() const
{
   return function_ptl;
}


void FunctorTypeDataItem::setReturnParams(std::list<C_parameter_type> *ptl)
{
   return_ptl = ptl;
}


std::list<C_parameter_type> * FunctorTypeDataItem::getReturnParams() const
{
   return return_ptl;
}


DataItem &FunctorTypeDataItem::assign(const DataItem &di)
{
   FunctorTypeDataItem const *f = dynamic_cast<FunctorTypeDataItem const*>(&di);
   if (f==0) {
      std::cerr<<"Failed dynamic cast to FunctorTypeDataItem!"<<std::endl;
      exit(-1);
   }
   _functorType = f->_functorType;
   constructor_ptl = f->constructor_ptl;
   function_ptl = f->function_ptl;
   return_ptl = f->return_ptl;
   category = f->category;
   return *this;
}
