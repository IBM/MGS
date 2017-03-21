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

#include "C_stride_list.h"
#include "C_order.h"
#include "C_steps.h"
#include "C_stride.h"
#include "SyntaxError.h"
#include "C_production.h"

void C_stride_list::internalExecute(LensContext *c)
{
   _cSteps->execute(c);
   _cStride->execute(c);
   _cOrder->execute(c);

   // Copy the lists over. The "=" operator copies the list
   _steps  = *_cSteps->getListInt();
   _strides = *_cStride->getListInt();
   _order  = *_cOrder->getListInt();

}

C_stride_list::C_stride_list(const C_stride_list& rv)
   : C_production(rv), _cSteps(0), _cStride(0), _cOrder(0)
{
   if (rv._cSteps) {
      _cSteps = rv._cSteps->duplicate();
   }
   if (rv._cStride) {
      _cStride = rv._cStride->duplicate();
   }
   if (rv._cOrder) {
      _cOrder = rv._cOrder->duplicate();
   }
}

C_stride_list::C_stride_list(C_steps *sp, C_stride *st,C_order *order, 
			     SyntaxError * error)
   : C_production(error), _cSteps(sp), _cStride(st), _cOrder(order)
{
}

const std::list<int>& C_stride_list::getStepsListInt() const
{
   return _steps;
}

const std::list<int>& C_stride_list::getStrideListInt() const
{
   return _strides;
}

const std::list<int>& C_stride_list::getOrderListInt() const
{
   return _order;
}

StridesList * C_stride_list::getStridesList()
{
   return this;
}

C_stride_list* C_stride_list::duplicate() const
{
   return new C_stride_list(*this);
}

C_stride_list::~C_stride_list()
{
   delete _cStride;
   delete _cSteps;
   delete _cOrder;
}

void C_stride_list::checkChildren() 
{
   if (_cSteps) {
      _cSteps->checkChildren();
      if (_cSteps->isError()) {
         setError();
      }
   }
   if (_cStride) {
      _cStride->checkChildren();
      if (_cStride->isError()) {
         setError();
      }
   }
   if (_cOrder) {
      _cOrder->checkChildren();
      if (_cOrder->isError()) {
         setError();
      }
   }
} 

void C_stride_list::recursivePrint() 
{
   if (_cSteps) {
      _cSteps->recursivePrint();
   }
   if (_cStride) {
      _cStride->recursivePrint();
   }
   if (_cOrder) {
      _cOrder->recursivePrint();
   }
   printErrorMessage();
} 
