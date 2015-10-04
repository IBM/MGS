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

#ifndef TOUCH_H
#define TOUCH_H

#include <mpi.h>
#include <vector>
#include <string.h>

#include "SegmentDescriptor.h"

#define LTWT_TOUCH

#ifndef LTWT_TOUCH
#define N_TOUCH_DATA 5
#else
#define N_TOUCH_DATA 4
#endif

class Touch
{
  friend class TissueContext;

 public:
  Touch();
  Touch(Touch const &);
  Touch& operator=(const Touch& touch);
  bool operator==(const Touch& touch);
  static MPI_Datatype* getTypeTouch();
  ~Touch();
		
  double* getTouchData() {return _touchData;}
  double* getKeys() {return _touchData;}
  double* getProps() {return &_touchData[2];}

  double getKey1() {return _touchData[0];}
  double getKey2() {return _touchData[1];}
  void setKey1(double key1) {_touchData[0]=key1;}
  void setKey2(double key2) {_touchData[1]=key2;}
  void setProp1(double prop1) {_touchData[2]=prop1;}
  void setProp2(double prop2) {_touchData[3]=prop2;}

  double getPartner(double key);

  void readFromFile(FILE* dataFile);
  void writeToFile(FILE* dataFile);
  void printTouch();

#ifndef LTWT_TOUCH
  const double getDistance() {return _touchData[4];}
  void setDistance(double distance) {_touchData[4]=distance;}

  short* getEndTouches() {return _endTouch;}
  short getEndTouch1() {return _endTouch[0];}
  short getEndTouch2() {return _endTouch[1];}
  short getEndTouch3() {return _endTouch[2];}
  short getEndTouch4() {return _endTouch[3];}

  bool remains() {return _remains;}
  void eliminate() {_remains=false;}
  void reinstate() {_remains=true;}
#endif

  class compare
    {
    public:
      compare(int c);
      bool operator()(const Touch& t0, const Touch& t1);
    private:
      int _case;
    };


  double getProp1() {return _touchData[2];}
  double getProp2() {return _touchData[3];}
  double getProp(double key);

 private:

  double _touchData[N_TOUCH_DATA];

#ifndef LTWT_TOUCH
  short _endTouch[4]; // end1A, end1B, end2A, end2B
  bool _remains;
#endif		

  static MPI_Datatype* _typeTouch;
  static SegmentDescriptor _segmentDescriptor;
};

#endif
