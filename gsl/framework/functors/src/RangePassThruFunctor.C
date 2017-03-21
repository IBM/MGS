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

#include "RangePassThruFunctor.h"
#include "FunctorType.h"
#include "NumericDataItem.h"
#include "LensContext.h"
//#include <iostream>
#include "DataItem.h"
#include "FloatDataItem.h"
#include "FunctorDataItem.h"
#include "StringDataItem.h"
#include "InstanceFactoryQueriable.h"
#include "DataItemQueriable.h"
#include "FunctorDataItem.h"
#include "SyntaxErrorException.h"

class Functor;
class FunctorType;
class Simulation;
class FloatDataItem;

void RangePassThruFunctor::doInitialize(LensContext *c, 
					const std::vector<DataItem*>& args)
{

   // get left_limit, testval, right_limit, left_oper and right_oper

   NumericDataItem* ndi = dynamic_cast<NumericDataItem*>(args[0]);
   if (ndi == 0) {
      throw SyntaxErrorException(
	 "Dynamic cast of DataItem to NumericDataItem failed in RangePassThruFunctor");
   }
   _left_limit = ndi->getFloat();

   StringDataItem* sdi = dynamic_cast<StringDataItem*>(args[1]);
   if (sdi == 0) {
      throw SyntaxErrorException(
	 "Dynamic cast of DataItem to StringDataItem failed in RangePassThruFunctor");
   }
   _left_oper = sdi->getString();

   FunctorDataItem* fdi = dynamic_cast<FunctorDataItem*>(args[2]);
   if (fdi == 0) {
      throw SyntaxErrorException(
	 "Dynamic cast of DataItem to FunctorDataItem failed in RangePassThruFunctor");
   }
   _testFunct = fdi->getFunctor();

   sdi = dynamic_cast<StringDataItem*>(args[3]);
   if (sdi == 0) {
      throw SyntaxErrorException(
	 "Dynamic cast of DataItem to StringDataItem failed in RangePassThruFunctor");
   }
   _right_oper = sdi->getString();

   ndi = dynamic_cast<NumericDataItem*>(args[4]);
   if (ndi == 0) {
      throw SyntaxErrorException(
	 "Dynamic cast of DataItem to NumericDataItem failed in RangePassThruFunctor");
   }
   _right_limit = ndi->getFloat();

}


void RangePassThruFunctor::doExecute(LensContext *c, 
				     const std::vector<DataItem*>& args, 
				     std::auto_ptr<DataItem>& rvalue)
{
   std::vector<DataItem*> nullArgs;
   std::auto_ptr<DataItem> rval_ap;
   FloatDataItem *fdi;

   double _numTestVal;

   if ( (_left_oper == "<=")  && (_right_oper == "<=") ) {
      if ( _left_limit >= _right_limit )
         std::cerr 
	    << "Warning: Possible mismatch of limits in RangePassThru" 
	    << std::endl;

      for ( ;; ) {
         _testFunct->execute(c, nullArgs, rval_ap);
         fdi = dynamic_cast<FloatDataItem*>(rval_ap.get());
         if ( fdi == 0 ) {
            throw SyntaxErrorException(
	       "Invalid cast to Float in RangePassThru");
         }
         _numTestVal = fdi->getFloat();
         if ( (_numTestVal >= _left_limit) && (_numTestVal <= _right_limit) )
            break;
      }
   }
   else if ( (_left_oper == "<")  && (_right_oper == "<=") ) {
      if ( _left_limit >= _right_limit )
         std::cerr << "Warning: Possible mismatch of limits in RangePassThru" << std::endl;

      for ( ;; ) {
         _testFunct->execute(c, nullArgs, rval_ap);
         fdi = dynamic_cast<FloatDataItem*>(rval_ap.get());
         if ( fdi == 0 ) {
            throw SyntaxErrorException("Invalid cast to Float in RangePassThru");
         }
         _numTestVal = fdi->getFloat();
         if ( (_numTestVal > _left_limit) && (_numTestVal <= _right_limit) )
            break;
      }
   }
   else if ( (_left_oper == "<")  && (_right_oper == "<") ) {
      if ( _left_limit >= _right_limit )
         std::cerr 
	    << "Warning: Possible mismatch of limits in RangePassThru" 
	    << std::endl;

      for ( ;; ) {
         _testFunct->execute(c, nullArgs, rval_ap);
         fdi = dynamic_cast<FloatDataItem*>(rval_ap.get());
         if ( fdi == 0 ) {
            throw SyntaxErrorException(
	       "Invalid cast to Float in RangePassThru");
         }
         _numTestVal = fdi->getFloat();
         if ( (_numTestVal > _left_limit) && (_numTestVal < _right_limit) )
            break;
      }
   }
   else if ( (_left_oper == "<=")  && (_right_oper == "<") ) {
      if ( _left_limit >= _right_limit )
         std::cerr 
	    << "Warning: Possible mismatch of limits in RangePassThru" 
	    << std::endl;

      for ( ;; ) {
         _testFunct->execute(c, nullArgs, rval_ap);
         fdi = dynamic_cast<FloatDataItem*>(rval_ap.get());
         if ( fdi == 0 ) {
            throw SyntaxErrorException(
	       "Invalid cast to Float in RangePassThru");
         }
         _numTestVal = fdi->getFloat();
         if ( (_numTestVal >= _left_limit) && (_numTestVal < _right_limit) )
            break;
      }
   }
   else if ( (_left_oper == ">=")  && (_right_oper == ">=") ) {
      if ( _left_limit <= _right_limit )
         std::cerr 
	    << "Warning: Possible mismatch of limits in RangePassThru" 
	    << std::endl;

      for ( ;; ) {
         _testFunct->execute(c, nullArgs, rval_ap);
         fdi = dynamic_cast<FloatDataItem*>(rval_ap.get());
         if ( fdi == 0 ) {
            throw SyntaxErrorException(
	       "Invalid cast to Float in RangePassThru");
         }
         _numTestVal = fdi->getFloat();
         if ( (_numTestVal <= _left_limit) && (_numTestVal >= _right_limit) )
            break;
      }
   }
   else if ( (_left_oper == ">")  && (_right_oper == ">=") ) {
      if ( _left_limit <= _right_limit )
         std::cerr 
	    << "Warning: Possible mismatch of limits in RangePassThru" 
	    << std::endl;

      for ( ;; ) {
         _testFunct->execute(c, nullArgs, rval_ap);
         fdi = dynamic_cast<FloatDataItem*>(rval_ap.get());
         if ( fdi == 0 ) {
            throw SyntaxErrorException(
	       "Invalid cast to Float in RangePassThru");
         }
         _numTestVal = fdi->getFloat();
         if ( (_numTestVal < _left_limit) && (_numTestVal >= _right_limit) )
            break;
      }
   }
   else if ( (_left_oper == ">")  && (_right_oper == ">") ) {
      if ( _left_limit <= _right_limit )
         std::cerr 
	    << "Warning: Possible mismatch of limits in RangePassThru" 
	    << std::endl;

      for ( ;; ) {
         _testFunct->execute(c, nullArgs, rval_ap);
         fdi = dynamic_cast<FloatDataItem*>(rval_ap.get());
         if ( fdi == 0 ) {
            throw SyntaxErrorException("Invalid cast to Float in RangePassThru");
         }
         _numTestVal = fdi->getFloat();
         if ( (_numTestVal < _left_limit) && (_numTestVal > _right_limit) )
            break;
      }
   }
   else if ( (_left_oper == ">=")  && (_right_oper == ">") ) {
      if ( _left_limit <= _right_limit )
         std::cerr 
	    << "Warning: Possible mismatch of limits in RangePassThru" 
	    << std::endl;

      for ( ;; ) {
         _testFunct->execute(c, nullArgs, rval_ap);
         fdi = dynamic_cast<FloatDataItem*>(rval_ap.get());
         if ( fdi == 0 ) {
            throw SyntaxErrorException(
	       "Invalid cast to Float in RangePassThru");
         }
         _numTestVal = fdi->getFloat();
         if ( (_numTestVal <= _left_limit) && (_numTestVal > _right_limit) )
            break;
      }
   }

   // We could add more alternatives; good enough for the time being.

   else {
      throw SyntaxErrorException(
	 "Inequality operators inconsistent in RangePassThru");
   }

   rvalue.reset(new FloatDataItem(*fdi));

}


void RangePassThruFunctor::duplicate(std::auto_ptr<Functor> &fap) const
{
   Functor *p = new RangePassThruFunctor(*this);
   fap.reset(p);
}


RangePassThruFunctor::RangePassThruFunctor()
{
}

RangePassThruFunctor::~RangePassThruFunctor()
{
}
