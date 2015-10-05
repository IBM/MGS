// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. and EPFL 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

#include "Touch.h"
#include "Utilities.h"
#include <cassert>
#include <iostream>
#include <float.h>

MPI_Datatype* Touch::_typeTouch = 0;

SegmentDescriptor Touch::_segmentDescriptor;

Touch::Touch() {
  for (int i = 0; i < N_TOUCH_DATA; ++i) {
    _touchData[i] = 0;
  }

#ifndef LTWT_TOUCH
  for (int i = 0; i < 4; ++i) {
    _endTouch[i] = 0;
  }
  _remains = true;
#endif
}

Touch::Touch(Touch const& t) {
  for (int i = 0; i < N_TOUCH_DATA; ++i) {
    _touchData[i] = t._touchData[i];
  }
#ifndef LTWT_TOUCH
  for (int i = 0; i < 4; ++i) {
    _endTouch[i] = t._endTouch[i];
  }
  _remains = t._remains;
#endif
}

MPI_Datatype* Touch::getTypeTouch() {
  if (_typeTouch == 0) {
    Touch t;
    _typeTouch = new MPI_Datatype;
#ifndef LTWT_TOUCH
    Datatype datatype(4, &t);
    datatype.set(0, MPI_LB, 0);
    datatype.set(1, MPI_DOUBLE, N_TOUCH_DATA, t._touchData);
    datatype.set(2, MPI_SHORT, 4, t._endTouch);
    datatype.set(3, MPI_UB, sizeof(Touch));
    *_typeTouch = datatype.commit();
#else
    Datatype datatype(3, &t);
    datatype.set(0, MPI_LB, 0);
    datatype.set(1, MPI_DOUBLE, N_TOUCH_DATA, t._touchData);
    datatype.set(2, MPI_UB, sizeof(Touch));
    *_typeTouch = datatype.commit();
#endif
  }
  return _typeTouch;
}

void Touch::readFromFile(FILE* dataFile) {
  size_t s = fread(_touchData, sizeof(double), N_TOUCH_DATA, dataFile);
#ifndef LTWT_TOUCH
  fread(_endTouch, sizeof(short), 4, dataFile);
#endif
}

void Touch::writeToFile(FILE* dataFile) {
  fwrite(_touchData, sizeof(double), N_TOUCH_DATA, dataFile);
#ifndef LTWT_TOUCH
  fwrite(_endTouch, sizeof(short), 4, dataFile);
#endif
}

void Touch::printTouch() {
  std::cerr << _segmentDescriptor.getNeuronIndex(_touchData[0]) << " "
            << _segmentDescriptor.getBranchIndex(_touchData[0]) << " "
            << _segmentDescriptor.getSegmentIndex(_touchData[0]) << " "
            << _segmentDescriptor.getNeuronIndex(_touchData[1]) << " "
            << _segmentDescriptor.getBranchIndex(_touchData[1]) << " "
            << _segmentDescriptor.getSegmentIndex(_touchData[1]) << " "
            << _touchData[2] << " " << _touchData[3] << " "
#ifndef LTWT_TOUCH
            << _touchData[4] << " " << _endTouch[0] << _endTouch[1]
            << _endTouch[2] << _endTouch[3] << " "
#endif
      ;
}

double Touch::getPartner(double key) {
  double rval = 0;
  if (key == getKey1())
    rval = getKey2();
  else if (key == getKey2())
    rval = getKey1();
  return rval;
}

double Touch::getProp(double key) {
  double rval = DBL_MAX;
  if (key == getKey1())
    rval = getProp1();
  else if (key == getKey2())
    rval = getProp2();
  return rval;
}

Touch::~Touch() {}

Touch::compare::compare(int c) : _case(c) {}

bool Touch::compare::operator()(const Touch& t0, const Touch& t1) {
  bool rval = false;

  double key0, key1, key2, key3;
  if (_case == 0) {
    key0 = t0._touchData[0];
    key1 = t1._touchData[0];
    key2 = t0._touchData[1];
    key3 = t1._touchData[1];
  } else if (_case == 1) {
    key0 = t0._touchData[1];
    key1 = t1._touchData[1];
    key2 = t0._touchData[0];
    key3 = t1._touchData[0];
  } else
    assert(0);

  unsigned int n0 = _segmentDescriptor.getNeuronIndex(key0);
  unsigned int n1 = _segmentDescriptor.getNeuronIndex(key1);

  if (n0 == n1) {
    unsigned int b0 = _segmentDescriptor.getBranchIndex(key0);
    unsigned int b1 = _segmentDescriptor.getBranchIndex(key1);

    if (b0 == b1) {
      unsigned int s0 = _segmentDescriptor.getSegmentIndex(key0);
      unsigned int s1 = _segmentDescriptor.getSegmentIndex(key1);

      if (s0 == s1) {
        unsigned int n2 = _segmentDescriptor.getNeuronIndex(key2);
        unsigned int n3 = _segmentDescriptor.getNeuronIndex(key3);

        if (n2 == n3) {
          unsigned int b2 = _segmentDescriptor.getBranchIndex(key2);
          unsigned int b3 = _segmentDescriptor.getBranchIndex(key3);

          if (b2 == b3) {
            unsigned int s2 = _segmentDescriptor.getSegmentIndex(key2);
            unsigned int s3 = _segmentDescriptor.getSegmentIndex(key3);

            rval = (s2 < s3);

          } else
            rval = (b2 < b3);

        } else
          rval = (n2 < n3);

      } else
        rval = (s0 < s1);

    } else
      rval = (b0 < b1);

  } else
    rval = (n0 < n1);

  return rval;
}

Touch& Touch::operator=(const Touch& t) {
  if (this == &t) return *this;
  std::copy(t._touchData, t._touchData + N_TOUCH_DATA, _touchData);
// memcpy(_touchData, t._touchData, N_TOUCH_DATA*sizeof(double));
#ifndef LTWT_TOUCH
  std::copy(t._endTouch, t._endTouch + 4, _endTouch);
  // memcpy(_endTouch, t._endTouch, 4*sizeof(short));
  _remains = t._remains;
#endif
  return *this;
}

bool Touch::operator==(const Touch& t) {
  if (this == &t) return true;
  double key0 = _touchData[0];
  double key1 = t._touchData[0];
  double key2 = _touchData[1];
  double key3 = t._touchData[1];
  return ((key0 == key1 && key2 == key3));  //|| (key0==key3 && key1==key2) );
}
