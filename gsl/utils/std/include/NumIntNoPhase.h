// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef NUMINTNOPHASE_H
#define NUMINTNOPHASE_H
#include "Copyright.h"


#include "ShallowArray.h"

class numIntNoPhase
{
 protected :

  virtual void initializeIterator(int, double) = 0;
  virtual void callIterate(ShallowArray< double > &) = 0;
  virtual void derivs(const ShallowArray< double > &, ShallowArray< double > &) = 0;

};


class EUNoPhase : public numIntNoPhase
{
protected:
  
  void initializeIterator(int, double);
  void callIterate(ShallowArray< double > &);

 private:

  ShallowArray< double > dx1;
  ShallowArray< double >::iterator i1, i2;
  double dT;

};



class RK4NoPhase : public numIntNoPhase
{
  // public:
  //  RK4(){} 
  
  
  protected:
  
  void initializeIterator(int, double);
  void callIterate(ShallowArray< double > &);

 private:

  ShallowArray< double > dx1, dx2, dx3, x1;
  ShallowArray< double >::iterator i1, i2, i3, i4;
  double dT, dT2, dT6;
};

#endif
