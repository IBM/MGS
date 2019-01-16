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

#ifndef NUMINT_H
#define NUMINT_H
#include "Copyright.h"


#include "ShallowArray.h"

#define DPREC 10.0e-12
class numInt
{
 protected :

  virtual void initializeIterator(int, double) = 0;
  virtual void callIteratePhase1() = 0;
  virtual void callIteratePhase2() = 0;
  virtual void callIteratePhase3() = 0;
  virtual void callIteratePhase4() = 0;
  virtual void prePhase1() = 0;
  virtual void prePhase2() = 0;
  virtual void prePhase3() = 0;
  virtual void prePhase4() = 0;
  virtual void derivs(const ShallowArray< double > &, ShallowArray< double > &) = 0;
  virtual void flushVars(const ShallowArray< double > &) = 0;

  //const ShallowArray< double > & Vars() const {return x;}
  //ShallowArray< double > & Vars() {return x;}
  //const double & deltaT() const {return dT;}
  //double & deltaT() {return dT;}

 protected:
  ShallowArray< double > x;
  double dT;
};

class EU : public numInt
{
protected:
  
  void initializeIterator(int, double);
  void callIteratePhase1();
  void callIteratePhase2(){}
  void callIteratePhase3(){}
  void callIteratePhase4(){}
  void prePhase1(){flushVars(x);}
  void prePhase2(){}
  void prePhase3(){}
  void prePhase4(){}
  //virtual void flushVars_x() = 0;
  //virtual void derivs(const ShallowArray< double > &, ShallowArray< double > &) = 0;

  

 private:

  ShallowArray< double > dx1;
  ShallowArray< double >::iterator i1, i2;

};



class RK4 : public numInt
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


class RK4Phased  : public numInt
{
  protected:
  
  void initializeIterator(int, double);
  void callIteratePhase1();
  void callIteratePhase2();
  void callIteratePhase3();
  void callIteratePhase4();
  void prePhase1(){flushVars(x);}
  void prePhase2(){flushVars(x1);}
  void prePhase3(){flushVars(x1);}
  void prePhase4(){flushVars(x1);}
  //virtual void derivs(const ShallowArray< double > &, ShallowArray< double > &) = 0;
  //virtual void flushVars_x() = 0;
  //virtual void flushVars_x1() = 0;

  //ShallowArray< double > x;

 private:

  ShallowArray< double > dx1, dx2, dx3, x1;
  ShallowArray< double >::iterator i1, i2, i3, i4;
  double dT2, dT6;
};



class RK4Counter : public RK4Phased
{
 protected:

  void initializeIterator(int n, double dt)
  {
    RK4Phased::initializeIterator(n,dt);
    counter = 0;
  }

  void callIterate()
  {
    if (counter == 2 || counter == 4 || counter == 6) flushVars(rk1.x1);
    else if (counter == 0) flushVars(rk1.x);
    else if (counter == 1) callIteratePhase1(rk1);
    else if (counter == 3) callIteratePhase2(rk1);
    else if (counter == 5) callIteratePhase3(rk1);
    else callIteratePhase4(rk1);
    counter ++;
    if (counter == 8) counter = 0;
  }

 private:
  int counter;

};
class LypIntCounter : public RK4Phased
{
 protected:

  void initializeIterator(int n, double dt)
  {
    RK4Phased::initializeIterator(n,dt);
    counter = 0;
    rk2.dx1.increaseSizeTo(n);
    rk2.dx2.increaseSizeTo(n);
    rk2.dx3.increaseSizeTo(n);
    rk2.x1.increaseSizeTo(n);  
    rk2.x.increaseSizeTo(n);  
    d0 = DPREC;
  }

  void callIterate()
  {
    if (counter == 2 || counter == 4 || counter == 6) flushVars(rk1.x1);
    else if (counter == 0) flushVars(rk1.x);
    else if (counter == 1) callIteratePhase1(rk1);
    else if (counter == 3) callIteratePhase2(rk1);
    else if (counter == 5) callIteratePhase3(rk1);
    else if (counter == 7) callIteratePhase4(rk1);
    else if (counter == 10 || counter == 12 || counter == 14) flushVars(rk2.x1);
    else if (counter == 8) flushVars(rk2.x);
    else if (counter == 9) callIteratePhase1(rk2);
    else if (counter == 11) callIteratePhase2(rk2);
    else if (counter == 13) callIteratePhase3(rk2);
    else if (counter == 15) callIteratePhase4(rk2);
    counter ++;
    if (counter == 16) counter = 0;
  }

  
  double getDiffSqr()
  {
    double sum = 0;
    ShallowArray< double >::iterator i2, i3;
    for (i2=rk1.x.begin(),i3=rk2.x.begin();i2!=rk1.x.end();i2++,i3++) 
      {double val = *i2 - *i3; sum+=val*val;}
    return sum;
  }
  

  virtual double diffLen() = 0;


 private:
  int counter;
  RK4PhasedVars rk2;

  double d0;

  
  void setPetVars() 
  {
    double val1 = d0/diffLen();
    ShallowArray< double >::iterator i1,i2;
    for (i1=rk1.x.begin(),i2=rk2.x.begin();i1!=rk1.x.end();i1++,i2++) 
      *i2 = *i1 + (*i2 - *i1)*val1;
  }
  

  RNG rng;


};



/*
class LypInt : public RK4Phased
{
 protected :

  void initializeIterator(int n, double dt);
  
  void callIteratePhase1();
  void callIteratePhase2();
  void callIteratePhase3();
  void callIteratePhase4();
  void prePhase1(){RK4Phased::prePhase1();flushVars(rk2.x);}
  void prePhase2(){RK4Phased::prePhase2();flushVars(rk2.x1);}
  void prePhase3(){RK4Phased::prePhase3();flushVars(rk2.x1);}
  void prePhase4(){RK4Phased::prePhase4();flushVars(rk2.x1);}


  
  double getDiffSqr() const
  {
    double sum = 0;
    ShallowArray< double >::iterator i2, i3;
    for (i2=origInt.Vars().begin(),i3=petInt.Vars().begin();i2!=origInt.Vars().end();i2++,i3++) 
      {double val = *i2 - *i3; sum+=val*val;}
    return sum;
  }


  
  double diffLen(){}

 private:



  double d0;

  void setPetVars() 
  {
    double val1 = d0/diffLen();
    ShallowArray< double >::iterator i1,i2;
    for (i1=origInt.Vars().begin(),i2=petInt.Vars().begin();i1!=origInt.Vars().end();i1++,i2++) 
      *i2 = *i1 + (*i2 - *i1)*val1;
  }

  RNG rng;

  RK4PhasedVars rk2;

};
*/


/*
#define DPREC 10.0e-12
//#define LYPLAG 1000
template<class NUMINT_T>
class LypInt2 : public numInt
{
 protected :

  void initializeIterator(int n, double dt) 
  {
    origInt.initializeIterator(n,dt);
    petInt.initializeIterator(n,dt);
    //diffvec.increaseSizeTo(n);
    d0 = DPREC;
    //lypskip = LYPLAG;
    //count = 0;
  }

  void callIteratePhase1() 
  {
    origInt.callIteratePhase1();
    petInt.callIteratePhase1();
  }

  void callIteratePhase2() 
  {
    origInt.callIteratePhase2();
    petInt.callIteratePhase2();
  }

  void callIteratePhase3() 
  {
    origInt.callIteratePhase3();
    petInt.callIteratePhase3();
  }

  void callIteratePhase4() 
  {
    origInt.callIteratePhase4();
    petInt.callIteratePhase4();
  }

  void prePhase1() 
  {
    origInt.prePhase1();
    petInt.prePhase1();
    setPetVars();
  }

  void prePhase2() 
  {
    origInt.prePhase2();
    petInt.prePhase2();
  }

  void prePhase3() 
  {
    origInt.prePhase3();
    petInt.prePhase3();
  }

  void prePhase4() 
  {
    origInt.prePhase4();
    petInt.prePhase4();
  }


  double getDiffSqr() const
  {
    double sum = 0;
    ShallowArray< double >::iterator i2, i3;
    for (i2=origInt.Vars().begin(),i3=petInt.Vars().begin();i2!=origInt.Vars().end();i2++,i3++) 
      {double val = *i2 - *i3; sum+=val*val;}
    return sum;
  }


  virtual void derivs(const ShallowArray< double > &, ShallowArray< double > &) = 0;
  virtual void flushVars(const ShallowArray< double > &) = 0;


  const ShallowArray< double > & Vars() const {return origInt.Vars();} 
  ShallowArray< double > & Vars() {return origInt.Vars();} 

  //const ShallowArray< double > & Vars() const {return x;}
  //ShallowArray< double > & Vars() {return x;}
  //const double & deltaT() const {return dT;}
  //double & deltaT() {return dT;}

  //virtual double diffLen() const =  0;
  double diffLen(){}

 private:

  NUMINT_T origInt, petInt;
  double d0;

  void setPetVars() 
  {
    double val1 = d0/diffLen();
    //if (count >= lypskip) lyp+=log(val1/deltaT());
    //count++; 
    ShallowArray< double >::iterator i1,i2;
    for (i1=origInt.Vars().begin(),i2=petInt.Vars().begin();i1!=origInt.Vars().end();i1++,i2++) 
      *i2 = *i1 + (*i2 - *i1)*val1;
  }

  RNG rng;
  //double lyp;
  //int lypskip;
  //int count;

};


	       */


#endif
