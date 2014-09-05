// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "Sigmoid.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>

// centerpoint, top, bottom, slope
Sigmoid::Sigmoid(float c, float t, float b, float m)
: _beta(c), _table_x1(0), _table_x2(0), _table_dx(0), _tableSize(0), _table(0)
{
   if (b>=t) {
      std::cerr<<"Bad sigmoid asymptotes: top must be greater than bottom!"<<std::endl;
      exit(-1);
   }
   if (m == 0) {
      std::cerr<<"Sigmoid must have non-zero slope!"<<std::endl;
      exit(-1);
   }

   _d = t-b;
   _s = b/_d;
   _alpha = (4.0*m)/_d;
}


Sigmoid::Sigmoid(float alpha, float beta)
: _alpha(alpha), _beta(beta), _s(0), _d(1), _table_x1(0), _table_x2(0), _table_dx(0), _tableSize(0), _table(0)
{
   if (_alpha == 0) {
      std::cerr<<"Sigmoid must have non-zero alpha!"<<std::endl;
      exit(-1);
   }
}


Sigmoid::Sigmoid(Sigmoid const & s)
: _alpha(s._alpha), _beta(s._beta), _s(s._s), _d(s._d), _table(s._table)
{
   if (_table != 0) {
      _table_x1 = s._table_x1;
      _table_x2 = s._table_x2;
      _table_dx = s._table_dx;
      _tableSize = s._tableSize;
      _table = new float[_tableSize];
      float* p = _table + _tableSize;
      float* t = s._table + _tableSize;
      do {
         *(--p) = *(--t);
      } while (p != _table);
   }
}


float Sigmoid::compute(float x)
{
   float y;
   if ( _table != 0 && (x >= _table_x1) && (x <= _table_x2) ) {
      float* ty = _table + int( (x-_table_x1)/_table_dx );
      y = *ty;
      y += ( ( fmod(x, _table_dx) / _table_dx ) * ( *(++ty) - y ) );
   }
   else
      y = ( _d / ( 1 + exp( -_alpha * ( x - _beta) ) ) ) + (_d * _s);
   return y;
}


void Sigmoid::tableize(float x1, float x2, float dx)
{
   if ( x2>x1 && dx>0 ) {
      delete _table;
      _tableSize = int( (x2-x1)/dx ) + 1;
      float* p = _table = new float[_tableSize];
      _table_dx = dx;
      float x =_table_x1 = x1;
      _table_x2 = x1 + ( (_tableSize-1) * dx);
      while (x<=_table_x2) {
         *p++ = ( _d / ( 1 + exp( -_alpha * ( x - _beta) ) ) ) + (_d * _s);
         x+=dx;
      }
   }
   else
      std::cerr<<"Bad tableization parameters for Sigmoid!"<<std::endl;
}


void Sigmoid::tableize()
{
   if ( _table_x1 != 0 && _table_x2 != 0 && _table_dx != 0 )
      tableize(_table_x1, _table_x2, _table_dx);
   else std::cerr<<"Must parameterize tableization of Sigmoid!"<<std::endl;
}


void Sigmoid::beta(float beta)
{
   _beta = beta;
   delete _table;
   _table = 0;
}


void Sigmoid::alpha(float alpha)
{
   if (alpha == 0) {
      std::cerr<<"Sigmoid must have non-zero slope(m)!"<<std::endl;
      exit(-1);
   }
   _alpha = alpha;
   delete _table;
   _table = 0;
}


void Sigmoid::centerpoint(float beta)
{
   _beta = beta;
   delete _table;;
   _table = 0;
}


void Sigmoid::top(float t)
{
   float b = _s*_d;
   float m = (_alpha*_d)/4.0;
   if (b>=t) {
      std::cerr<<"Bad sigmoid asymptotes: top(t) must be greater than bottom!"<<std::endl;
      exit(-1);
   }
   _d = t-b;
   _s = b/_d;
   _alpha = (4.0*m)/_d;

   delete _table;
   _table = 0;
}


void Sigmoid::bottom(float b)
{
   float old_bottom = _s*_d;
   float t = _d+old_bottom;
   float m = (_alpha*_d)/4.0;
   if (b>=t) {
      std::cerr<<"Bad sigmoid asymptotes: top must be greater than bottom(b)!"<<std::endl;
      exit(-1);
   }

   _d = t-b;
   _s = b/_d;
   _alpha = (4.0*m)/_d;

   delete _table;
   _table = 0;
}


void Sigmoid::slope(float m)
{
   if (m == 0) {
      std::cerr<<"Sigmoid must have non-zero slope(m)!"<<std::endl;
      exit(-1);
   }
   _alpha = (4.0*m)/_d;

   delete _table;
   _table = 0;
}


bool Sigmoid::isTableized()
{
   return (_table != 0);
}


Sigmoid::~Sigmoid()
{
   delete _table;
}
