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

#ifndef SIGMOID_H
#define SIGMOID_H
#include "Copyright.h"

#include <string>

class Sigmoid
{
   private:
      float _alpha;              // sigmoid stiffness
      float _beta;               // sigmoid centerpoint
      float _s;                  // sigmoid y offset
      float _d;                  // sigmoid scaling factor
      float _table_x1;
      float _table_x2;
      float _table_dx;
      int _tableSize;
      float* _table;

   public:
                                 // centerpoint, top, bottom, slope
      Sigmoid(float c, float t, float b, float m);
                                 // stiffness, centerpoint (default top = 1, bottom = 0)
      Sigmoid(float alpha, float beta);
      Sigmoid(Sigmoid const & s);
      float compute(float x);
      void tableize(float x1, float x2, float dx);
      void tableize();           // for retableization
      bool isTableized();
      void alpha(float alpha);
      void beta(float beta);
      void centerpoint(float c);
      void top(float t);
      void bottom(float b);
      void slope(float m);
      ~Sigmoid();
};

#endif
