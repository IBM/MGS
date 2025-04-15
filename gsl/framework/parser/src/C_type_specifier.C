// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_type_specifier.h"
#include "C_parameter_type_pair.h"
#include "C_init_attr_type_node.h"
#include "C_init_attr_type_edge.h"
#include "C_matrix_type_specifier.h"
#include "C_initializable_type_specifier.h"
#include "C_non_initializable_type_specifier.h"
#include "C_matrix_type_specifier.h"
#include "C_argument_list.h"
#include "ParameterSet.h"
#include "ParameterSetDataItem.h"
#include "DataItem.h"
#include "DataItemArrayDataItem.h"
#include "FloatDataItem.h"
#include "FloatArrayDataItem.h"
#include "IntDataItem.h"
#include "IntArrayDataItem.h"
#include "CustomStringDataItem.h"
#include "NDPairDataItem.h"
#include "GridSetDataItem.h"
#include "NodeSetDataItem.h"
#include "EdgeSetDataItem.h"
#include "FunctorDataItem.h"
#include "RelativeNodeSetDataItem.h"
#include "RepertoireDataItem.h"
#include "ServiceDataItem.h"
#include "GranuleMapperDataItem.h"
#include "TriggerDataItem.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"
#include "C_production.h"
#include <stdio.h>
#include <string>

void C_type_specifier::internalExecute(LensContext *c) {
  if (_initTypeSpec) {
    _initTypeSpec->execute(c);
    _nextTypeSpec = _initTypeSpec->getNextTypeSpecifier();
  }
  if (_nonInitTypeSpec) {
    _nonInitTypeSpec->execute(c);
    _nextTypeSpec = _nonInitTypeSpec->getNextTypeSpecifier();
  }
}

DataItem *C_type_specifier::getValidArgument(DataItem *DI) {
  if (_initTypeSpec) {  // initializable
    if (_type == _PSET) {
      ParameterSetDataItem *parSetDI = dynamic_cast<ParameterSetDataItem *>(DI);
      if (!parSetDI) return 0;

      ParameterSet *parSet = parSetDI->getParameterSet();

      C_parameter_type_pair *parTypePair =
          _initTypeSpec->getParameterTypePair();
      if (parTypePair) {
        C_init_attr_type_node *initAttrTypeNode =
            parTypePair->getInitAttrTypeNode();
        C_init_attr_type_edge *initAttrTypeEdge =
            parTypePair->getInitAttrTypeEdge();

        if (initAttrTypeNode &&
            (parSet->getModelType() == initAttrTypeNode->getModelType()))
          return parSetDI;
        else if (initAttrTypeEdge)
          return parSetDI;
        else
          return 0;
      } else {
        // Ask the author for description
        std::string mes = "Unknown problem";
        throwError(mes);
      }
    } else if (_type == _LIST) {
      Type next;
      if (_nextTypeSpec) {
        next = _nextTypeSpec->getType();
      } else {
        next = _UNSPECIFIED;
      }
      if (next == _INT) {
        // verify each element is an IntDataItem
        // if successful, create IntArrayDataItem
        // if not successful, report error and exit
        DataItemArrayDataItem *ddi = dynamic_cast<DataItemArrayDataItem *>(DI);
        if (ddi) {
          std::vector<DataItem *>::iterator iter,
              begin = ddi->getModifiableDataItemVector()->begin(),
              end = ddi->getModifiableDataItemVector()->end();
          std::vector<int> coord(1);
          coord[0] = ddi->getModifiableDataItemVector()->size();
          IntArrayDataItem *idi = new IntArrayDataItem(coord);
          coord[0] = 0;
          for (iter = begin; iter != end; ++iter) {
            if ((*iter)->getType() == IntDataItem::_type) {
              int int_dat = dynamic_cast<IntDataItem*>(*iter)->getInt();
              idi->setInt(coord, int_dat);
            } else {
              std::string mes = "type of element in list of ints is not an int";
              throwError(mes);
            }
            coord[0]++;
          }
          return idi;
        } else {
          std::string mes = "cast to ArrayDataItem in list failed";
          throwError(mes);
        }
      } else if (next == _FLOAT) {
        // verify each element is a FloatDataItem
        // if successful, create FloatArrayDataItem
        // else if it is int, cast it to float,
        // if not successful, report error and exit
        DataItemArrayDataItem *ddi = dynamic_cast<DataItemArrayDataItem *>(DI);
        if (ddi) {
          std::vector<DataItem *>::iterator iter,
              begin = ddi->getModifiableDataItemVector()->begin(),
              end = ddi->getModifiableDataItemVector()->end();
          std::vector<int> coord(1);
          coord[0] = ddi->getModifiableDataItemVector()->size();
          FloatArrayDataItem *fdi = new FloatArrayDataItem(coord);
          coord[0] = 0;
          for (iter = begin; iter != end; ++iter) {
            const std::string& typeInfo = (*iter)->getType();
            if (typeInfo == FloatDataItem::_type) {
              fdi->setFloat(coord,
                            dynamic_cast<FloatDataItem*>(*iter)->getFloat());
            } else if (typeInfo == IntDataItem::_type) {
				//NOTE: Int is casted to Float
  				fdi->setFloat(coord,
                            dynamic_cast<FloatDataItem *>(*iter)->getFloat());
			}else {
              std::string mes =
                  "type of element in list of floats is not an float";
              throwError(mes);
            }
            coord[0]++;
          }
          return fdi;
        } else {
          std::string mes = "cast to ArrayDataItem in list failed";
          throwError(mes);
        }
      } else {
        // check for consistent INTs or FLOATs and construct
        // IntArrayDataItem or FloatArrayDataItem
        // otherwise construct DataItemArrayDataItem from
        // data items returned by calling getValidArgument on
        // each element of the original list

        DataItemArrayDataItem *ddi = dynamic_cast<DataItemArrayDataItem *>(DI);
        if (ddi) {
          std::vector<DataItem *>::iterator iter,
              begin = ddi->getModifiableDataItemVector()->begin(),
              end = ddi->getModifiableDataItemVector()->end();
          std::vector<int> coord(1);
          coord[0] = ddi->getModifiableDataItemVector()->size();
          bool checkint = true;
          bool checkfloat = true;
          for (iter = begin; iter != end; ++iter) {
            const std::string& typeInfo = (*iter)->getType();
            if (typeInfo != IntDataItem::_type) checkint = false;
            if ((typeInfo != FloatDataItem::_type) &&
                (typeInfo != IntDataItem::_type))
              checkfloat = false;
          }
          if (checkint) {
            // It's a pure int.
            IntArrayDataItem *idi = new IntArrayDataItem(coord);
            coord[0] = 0;
            for (iter = begin; iter != end; ++iter) {
              idi->setInt(coord, dynamic_cast<IntDataItem *>(*iter)->getInt());
              coord[0]++;
            }
            return idi;
          } else if (checkfloat) {
            // It's a pure float or a mix of floats and ints.
            // then cast all to float type
            FloatArrayDataItem* fdi = new FloatArrayDataItem(coord);
            coord[0] = 0;
            for (iter = begin; iter != end; ++iter) {
              if ((*iter)->getType() == IntDataItem::_type)
                fdi->setFloat(
                    coord, (float)dynamic_cast<IntDataItem *>(*iter)->getInt());
              else
                fdi->setFloat(coord,
                              dynamic_cast<FloatDataItem *>(*iter)->getFloat());
              coord[0]++;
            }
            return fdi;
          } else {
            // It's a mix of many things...
            return ddi;
            // Commented out, looks right but doesn't work, this should be fixed
            // for
            // lists inside lists  does not look right - sgc
            //                   DataItemArrayDataItem *rddi = new
            //                   DataItemArrayDataItem(coord);
            //                   for ( iter = begin; iter != end; ++iter ) {
            //                      std::unique_ptr<DataItem> *di_ap = new
            //                      std::unique_ptr<DataItem>(getValidArgument(*iter));
            //                      rddi->setDataItem ( coord, *di_ap );
            //                      coord[0]++;
            //                   }
            //                   return rddi;
          }
        } else {
          std::string mes = "cast to ArrayDataItem in list failed";
          throwError(mes);
        }
      }
    } else {  // for anything else
      // std::cout << "***Entering  else***"<<std::endl;
      // std::cout << "***DI->getType()="<<DI->getType()<<std::endl;
      // std::cout << "***getDataItemType()="<<getDataItemType()<<std::endl;
      if (DI->getType() != getDataItemType()) {
        // std::cout <<"***Returning null***"<<std::endl;
        return 0;

      } else {
        // std::cout <<"***Returning original dataitem***"<<std::endl;
        return DI;
      }
    }
  }

  // non initializable
  else if (_nonInitTypeSpec) {
    // std::cout << "***Entering  _nonInitTypeSpec***"<<std::endl;
    if (DI->getType() != getDataItemType())
      return 0;

    else
      return DI;
  } else {  // we have a problem
    std::string mes = "unknown problem";
    throwError(mes);
  }
  return 0;
}

C_type_specifier::C_type_specifier(const C_type_specifier &rv)
    : C_production(rv),
      _initTypeSpec(0),
      _nonInitTypeSpec(0),
      _type(rv.getType()),
      _nextTypeSpec(0) {
  if (rv._initTypeSpec) {
    _initTypeSpec = rv._initTypeSpec->duplicate();
    _nextTypeSpec = _initTypeSpec->getNextTypeSpecifier();
  }

  if (rv._nonInitTypeSpec) {
    _nonInitTypeSpec = rv._nonInitTypeSpec->duplicate();
    _nextTypeSpec = _nonInitTypeSpec->getNextTypeSpecifier();
  }
}

C_type_specifier::C_type_specifier(C_initializable_type_specifier *i,
                                   SyntaxError *error)
    : C_production(error),
      _initTypeSpec(i),
      _nonInitTypeSpec(0),
      _type(i->getType()),
      _nextTypeSpec(0) {}

C_type_specifier::C_type_specifier(C_non_initializable_type_specifier *n,
                                   SyntaxError *error)
    : C_production(error),
      _initTypeSpec(0),
      _nonInitTypeSpec(n),
      _type(n->getType()),
      _nextTypeSpec(0) {}

C_type_specifier *C_type_specifier::duplicate() const {
  return new C_type_specifier(*this);
}

C_type_specifier::~C_type_specifier() {
  delete _initTypeSpec;
  delete _nonInitTypeSpec;
}

C_type_specifier::Type C_type_specifier::getType() const { return _type; }

const char *C_type_specifier::getDataItemType() {
  const char *retval = 0;
  switch (_type) {
    case (_FLOAT):
      retval = FloatDataItem::_type;
      break;
    case (_INT):
      retval = IntDataItem::_type;
      break;
    case (_MATRIX): {
      std::string mes = "not implemented";
      throwError(mes);
    }
    case (_STRING):
      retval = CustomStringDataItem::_type;
      break;
    case (_NDPAIR):
      retval = NDPairDataItem::_type;
      break;
    // case(_GRIDCOORD): retval = GridSetDataItem::_type; break;
    case (_GRIDCOORD):
      retval = NodeSetDataItem::_type;
      break;
    case (_NODESET):
      retval = NodeSetDataItem::_type;
      break;
    case (_EDGETYPE):
      retval = EdgeSetDataItem::_type;
      break;
    case (_FUNCTOR):
      retval = FunctorDataItem::_type;
      break;
    case (_TRIGGER):
      retval = TriggerDataItem::_type;
      break;
    case (_PSET):
      retval = ParameterSetDataItem::_type;
      break;
    case (_RELNODESET):
      retval = RelativeNodeSetDataItem::_type;
      break;
    case (_LIST): {
      std::string mes = "not implemented";
      throwError(mes);
    }
    case (_REPNAME):
      retval = RepertoireDataItem::_type;
      break;
    case (_GRID):
      retval = RepertoireDataItem::_type;
      break;
    case (_COMPOSITE):
      retval = RepertoireDataItem::_type;
      break;
    case (_SERVICE):
      retval = ServiceDataItem::_type;
      break;
    case (_EDGESET):
      retval = EdgeSetDataItem::_type;
      break;
    case (_PORT):
      retval = IntDataItem::_type;
      break;
    case (_GRANULEMAPPER):
      retval = GranuleMapperDataItem::_type;
      break;
    default: {
      std::string mes = "wrong type";
      throwError(mes);
    }
  }
  return retval;
}

void C_type_specifier::checkChildren() {
  if (_initTypeSpec) {
    _initTypeSpec->checkChildren();
    if (_initTypeSpec->isError()) {
      setError();
    }
  }
  if (_nonInitTypeSpec) {
    _nonInitTypeSpec->checkChildren();
    if (_nonInitTypeSpec->isError()) {
      setError();
    }
  }
  if (_nextTypeSpec) {
    _nextTypeSpec->checkChildren();
    if (_nextTypeSpec->isError()) {
      setError();
    }
  }
}

void C_type_specifier::recursivePrint() {
  if (_initTypeSpec) {
    _initTypeSpec->recursivePrint();
  }
  if (_nonInitTypeSpec) {
    _nonInitTypeSpec->recursivePrint();
  }
  if (_nextTypeSpec) {
    _nextTypeSpec->recursivePrint();
  }
  printErrorMessage();
}
