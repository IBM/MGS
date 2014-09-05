#ifndef TissueElement_H
#define TissueElement_H

class TissueFunctor;

class TissueElement
{
 public:
  virtual void setTissueFunctor(TissueFunctor*) =0;
};

#endif
