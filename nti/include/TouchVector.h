// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

/*
* TouchVector.h
*
*  Created on: Feb 10, 2010
*      Author: wagnerjo
*/

#ifndef TOUCHVECTOR_H_
#define TOUCHVECTOR_H_
#define DEFAULT_TOUCH_BLOCK_SIZE 1000
#include <mpi.h>
#include <map>
#include <list>
#include "BlockVector.h"
#include "Touch.h"
#include "Capsule.h"

typedef BlockVector<Touch>::Index TouchIndex;

class TouchVector : public BlockVector<Touch>
{
  private:
  std::map<int, std::list<TouchIndex> > _touchMap;

  public:
  TouchVector(int defaultBlockSize = DEFAULT_TOUCH_BLOCK_SIZE)
      : BlockVector<Touch>(defaultBlockSize), fieldLastEnd(this, 0, 0)
  {
  }
  virtual ~TouchVector() {}

  // Returns pointer to first touch of first block...
  Touch *getTouchOrigin()
  {
    return (getBlockCount() == 0 ? 0 : &getValue(0)->getValue(0));
  }
  // Delete all blocks and clear the touch map...
  void clear();
  void mapTouch(int i, TouchIndex ti);
  std::map<int, std::list<TouchIndex> > &getTouchMap() { return _touchMap; }
  //
  class TouchIterator : public BlockVector<Touch>::Index
  {
private:
    TouchVector *fieldTouchVector;

public:
    TouchIterator(TouchVector *touchVector, int block, int index)
        : BlockVector<Touch>::Index(block, index)
    {
      fieldTouchVector = touchVector;
    }
    TouchIterator(const TouchIterator &touchIterator)
        : BlockVector<Touch>::Index(touchIterator)
    {
      fieldTouchVector = touchIterator.getTouchVector();
    }
    virtual ~TouchIterator() {}

    TouchIterator &operator=(const TouchIterator &touchIterator)
    {
      BlockVector<Touch>::Index::operator=(touchIterator);
      fieldTouchVector = touchIterator.getTouchVector();
      return (*this);
    }
    TouchVector *getTouchVector() const { return (fieldTouchVector); }
    Touch *getValue() const { return (&(getTouchVector()->getValue(*this))); }
    bool operator!=(const TouchIterator &touchIterator) const
    {
      bool base = BlockVector<Touch>::Index::operator!=(touchIterator);
      return (base || getTouchVector() != touchIterator.getTouchVector());
    }
    Touch &operator*() { return (*(getValue())); }
    const Touch &operator*() const { return (*(getValue())); }
    Touch *operator->() { return (getValue()); }
    const Touch *operator->() const { return (getValue()); }
	// GOAL: provide a way to perform increment (+1), and the current variable
	//       keep the new reference
    TouchIterator &operator++()
    {
      Block<Touch> *block = getTouchVector()->getValue(getBlock());
      if (getIndex() + 1 < block->getCount())
      {
        setIndex(getIndex() + 1);
      }
      else
      {
        setBlock(getBlock() + 1);
        setIndex(0);
      }
      return (*this);
    }
	// GOAL: provide a way to perform increment (+1), but transfer the reference to 
	//       another variable
    TouchIterator &incrementOne()
    {
      Block<Touch> *block = getTouchVector()->getValue(getBlock());
      if (getIndex() + 1 < block->getCount())
      {
        setIndex(getIndex() + 1);
      }
      else
      {
        setBlock(getBlock() + 1);
        setIndex(0);
      }
      return (*this);
    }
    /*  TouchIterator &operator+(int i) {
                  Block<Touch> *block = getTouchVector()->getValue(getBlock());
                  do
                  {
                          if (getIndex() + i < block->getCount()) {
                                  setIndex(getIndex() + i);
                          } else {
                                  setBlock(getBlock() + 1);
                                  i = i - (block->getCount() - getIndex() - 1);
                                  setIndex(0);
                          }
                  }while (i > block->getCount());

                  if (getIndex() + i < block->getCount()) {
                          setIndex(getIndex() + i);
                  } else {
                          setBlock(getBlock() + 1);
                          setIndex(0);
                  }
                  return(*this);
      }
          */
  };
  //
  TouchIterator begin();
  TouchIterator end();
  TouchIterator begin(Capsule &capsule, int direction);
  TouchIterator end(Capsule &capsule, int direction);
  void sort(Touch::compare &c);
  bool unique();

  private:
  void heapSort(Touch::compare &c);
  void demote(Touch::compare &c, int boss, int topEmployee);
  void bubbleSort(Touch::compare &c);
  TouchIterator fieldLastEnd;
};

#endif /* TOUCHVECTOR_H_ */
