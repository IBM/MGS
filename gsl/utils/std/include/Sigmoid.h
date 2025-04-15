// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
