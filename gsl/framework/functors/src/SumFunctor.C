// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "SumFunctor.h"
#include "FunctorType.h"
#include "NumericDataItem.h"
#include "LensContext.h"
//#include <iostream>
#include "DataItem.h"
#include "IntDataItem.h"
#include "FloatDataItem.h"
#include "IntArrayDataItem.h"
#include "FloatArrayDataItem.h"
#include "DataItemArrayDataItem.h"
#include "InstanceFactoryQueriable.h"
#include "DataItemQueriable.h"
#include "FunctorDataItem.h"
#include "SyntaxErrorException.h"
#include <stdio.h>
#include <string.h>

class FunctorType;
class Simulation;

void SumFunctor::doInitialize(LensContext *c,
                              const std::vector<DataItem *> &args) {}

void SumFunctor::doExecute(LensContext *c, const std::vector<DataItem *> &args,
                           std::unique_ptr<DataItem> &rvalue) {

  std::cout << "DataItem type: " << args[0]->getType() << std::endl;

  DataItemArrayDataItem *ddi = dynamic_cast<DataItemArrayDataItem *>(args[0]);
  IntArrayDataItem *idi = dynamic_cast<IntArrayDataItem *>(args[0]);
  FloatArrayDataItem *fdi = dynamic_cast<FloatArrayDataItem *>(args[0]);

  if (idi) {
    _int_array->clear();
    std::vector<int>::iterator iter,
        begin = idi->getModifiableIntVector()->begin(),
        end = idi->getModifiableIntVector()->end();
    for (iter = begin; iter != end; ++iter) _int_array->push_back(*iter);
    _float_array = 0;
    _di_array = 0;
  } else if (fdi) {
    _float_array->clear();
    std::vector<float>::iterator iter,
        begin = fdi->getModifiableFloatVector()->begin(),
        end = fdi->getModifiableFloatVector()->end();
    for (iter = begin; iter != end; ++iter) _float_array->push_back(*iter);
    _int_array = 0;
    _di_array = 0;
  } else if (ddi) {
    _di_array->clear();
    std::vector<DataItem *>::iterator iter,
        begin = ddi->getModifiableDataItemVector()->begin(),
        end = ddi->getModifiableDataItemVector()->end();
    for (iter = begin; iter != end; ++iter) _di_array->push_back(*iter);
    _float_array = 0;
    _int_array = 0;
  } else {
    throw SyntaxErrorException(
        "Dynamic cast of DataItem to Int or Float ArrayDataItem failed in "
        "SumFunctor");
  }

  bool checkint = true;

  if (_int_array) {
    int _i_total = 0;
    std::vector<int>::iterator iter, begin = _int_array->begin(),
                                     end = _int_array->end();
    for (iter = begin; iter != end; ++iter) _i_total += *iter;
    IntDataItem *i_di = new IntDataItem;
    i_di->setInt(_i_total);
    rvalue.reset(i_di);
    std::cout << "Executed SumFunctor, returning int " << i_di->getInt() << "."
              << std::endl;
  } else if (_float_array) {
    float _f_total = 0.0;
    std::vector<float>::iterator iter, begin = _float_array->begin(),
                                       end = _float_array->end();
    for (iter = begin; iter != end; ++iter) _f_total += *iter;
    FloatDataItem *f_di = new FloatDataItem;
    f_di->setFloat(_f_total);
    rvalue.reset(f_di);
    std::cout << "Executed SumFunctor, returning float " << f_di->getFloat()
              << "." << std::endl;
  } else if (_di_array) {
    float total = 0.0;
    std::vector<DataItem *>::iterator iter, begin = _di_array->begin(),
                                            end = _di_array->end();
    for (iter = begin; iter != end; ++iter) {
      const std::string& typeInfo = (*iter)->getType();
      if (typeInfo == IntDataItem::_type)
        total += (float)dynamic_cast<IntDataItem*>(*iter)->getInt();
      else if (typeInfo == FloatDataItem::_type) {
        total += dynamic_cast<FloatDataItem*>(*iter)->getFloat();
        checkint = false;
      } else {
        throw SyntaxErrorException(
            "Cannot add non-numeric items in SumFunctor");
      }
    }
    if (checkint) {
      IntDataItem *i_di = new IntDataItem;
      i_di->setInt((int)total);
      rvalue.reset(i_di);
      std::cout << "Executed SumFunctor, returning -int " << (int)total << "."
                << std::endl;
    } else {
      FloatDataItem *f_di = new FloatDataItem;
      f_di->setFloat(total);
      rvalue.reset(f_di);
      std::cout << "Executed SumFunctor, returning -float " << total << "."
                << std::endl;
    }
  } else {
    throw SyntaxErrorException("Something wrong in SumFunctor");
  }
}

void SumFunctor::duplicate(std::unique_ptr<Functor> &fap) const {
  Functor *p = new SumFunctor(*this);
  fap.reset(p);
}

SumFunctor::SumFunctor() {
  _float_array = new std::vector<float>;
  _int_array = new std::vector<int>;
  _di_array = new std::vector<DataItem *>;
}

SumFunctor::SumFunctor(const SumFunctor &f) {
  _float_array = new std::vector<float>(*f._float_array);

  _int_array = new std::vector<int>(*f._int_array);

  std::unique_ptr<DataItem> diap;
  _di_array = new std::vector<DataItem *>(f._di_array->size());
  std::vector<DataItem *>::const_iterator i, begin = f._di_array->begin();
  std::vector<DataItem *>::const_iterator end = f._di_array->end();
  std::vector<DataItem *>::iterator j = _di_array->begin();
  for (i = begin; i != end; ++i) {
    (*i)->duplicate(diap);
    (*j) = diap.release();
  }
}

SumFunctor::~SumFunctor() {
  std::vector<DataItem *>::const_iterator i, begin = _di_array->begin();
  std::vector<DataItem *>::const_iterator end = _di_array->end();
  for (i = begin; i != end; ++i) {
    delete *i;
  }
  delete _float_array;
  delete _int_array;
  delete _di_array;
}
