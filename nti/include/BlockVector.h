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
 * BlockVector.h
 *
 *  Created on: Feb 13, 2010
 *      Author: wagnerjo
 */

#ifndef BLOCKVECTOR_H_
#define BLOCKVECTOR_H_

#include <mpi.h>
#include <vector>
#include <cassert>
#include "Block.h"

#ifndef DISABLE_PTHREADS
#include "For.h"
#endif

class Mutex;

template<typename T> class BlockVector {
 private:
  int fieldDefaultBlockSize;
  std::vector<Block<T> *> fieldBlocks;
 public:
  BlockVector(int defaultBlockSize = 10000) {
    assert(defaultBlockSize > 0);
    fieldDefaultBlockSize = defaultBlockSize;
  }
  BlockVector(const BlockVector &blockVector) {
    fieldDefaultBlockSize = blockVector.getDefaultBlockSize();
    for (int i = 0; i < blockVector.getBlockCount(); i++) {
      fieldBlocks.push_back(new Block<T>(blockVector[i]));
    }
  }
  virtual ~BlockVector() {
    for (int i = 0; i < getBlockCount(); i++) {
      delete fieldBlocks[i];
    }
  }
  void merge(BlockVector& blockVector) {
    for (int i = 0; i < blockVector.getBlockCount(); ++i) push_back(blockVector.fieldBlocks[i]);
    blockVector.fieldBlocks.clear();
  }
  void clear() {
    for (int i = 0; i < getBlockCount(); i++) delete fieldBlocks[i];
    fieldBlocks.clear();
  }

  int getCount() {
    int rval=0;
    for (int i = 0; i < getBlockCount(); i++) rval+=fieldBlocks[i]->getCount();
    return rval;
  }

  int getDefaultBlockSize() const {
    return(fieldDefaultBlockSize);
  }
  int getBlockCount() const {
    return(fieldBlocks.size());
  }
  int push_back(Block<T> *block) {
	assert(block != 0);
    fieldBlocks.push_back(block);
    return(getBlockCount() - 1);
  }
  Block<T> *&getValue(int i) {
    assert(0 <= i && i < getBlockCount());
    return(fieldBlocks[i]);
  }
  const Block<T> *&getValue(int i) const {
    assert(0 <= i && i < getBlockCount());
    return(fieldBlocks[i]);
  }
  void setValue(int i, Block<T> *&block) {
    assert(0 <= i && i < getBlockCount());
    delete fieldBlocks[i];
    fieldBlocks[i] = block;
  }
  Block<T>* getBlock(int i) {
    assert(0 <= i && i < getBlockCount());
    return(fieldBlocks[i]);
  }
  const Block<T>* getBlock(int i) const {
    assert(0 <= i && i < getBlockCount());
    return(fieldBlocks[i]);
  }
  //
  class Index {
  private:
    int fieldBlock;
    int fieldIndex;
  public:
    Index() : fieldBlock(0), fieldIndex(0) {}
    Index(int block, int index) {
      fieldBlock = block;
      fieldIndex = index;
    }
    Index(const Index &index) {
      fieldBlock = index.getBlock();
      fieldIndex = index.getIndex();
    }
    virtual ~Index() {}
    Index &operator=(const Index &index) {
      fieldBlock = index.getBlock();
      fieldIndex = index.getIndex();
      return(*this);
    }
    bool operator==(const Index &index) const {
        return(getBlock() == index.getBlock() && getIndex() == index.getIndex());
    }
    bool operator!=(const Index &index) const {
        return(getBlock() != index.getBlock() || getIndex() != index.getIndex());
    }
    int getBlock() const {
      return(fieldBlock);
    }
    void setBlock(int block) {
      fieldBlock = block;
    }
    int getIndex() const {
      return(fieldIndex);
    }
    void setIndex(int index) {
      fieldIndex = index;
    }
  };

  Index push_back(const T &t, Mutex* mutex)
{
    int i = getBlockCount();
    if (i == 0 || getValue(i-1)->getCount() >= getValue(i-1)->getSize()) {
#ifndef DISABLE_PTHREADS
      if (mutex) mutex->lock();
#endif
      push_back(new Block<T>(getDefaultBlockSize()));
#ifndef DISABLE_PTHREADS
      if (mutex) mutex->unlock();
#endif
      assert(getBlockCount() == i + 1);
    }
    assert(getBlockCount() == fieldBlocks.size());
    return(Index(getBlockCount() - 1, getValue(getBlockCount() - 1)->push_back(t)));
  }

  T &getValue(Index index) {
    assert (0 <= index.getBlock() && index.getBlock() < getBlockCount());
    getValue(index.getBlock())->getValue(index.getIndex());
    return(getValue(index.getBlock())->getValue(index.getIndex()));
  }
  const T &getValue(Index index) const {
    assert(0 <= index.getBlock() && index.getBlock() < getBlockCount());
    getValue(index.getBlock())->getValue(index.getIndex());
    return(getValue(index.getBlock())->getValue(index.getIndex()));
  }
  void setValue(Index index, T &t) {
    assert(0 <= index.getBlock() && index.getBlock() < getBlockCount());
    getValue(index.getBlock())->getValue(index.getIndex());
    getValue(index.getBlock())->setValue(index.getIndex(), t);
  }
  T &operator[](Index index) {
    assert(0 <= index.getBlock() && index.getBlock() < getBlockCount());
    getValue(index.getBlock())->getValue(index.getIndex());
    return(getValue(index.getBlock())->getValue(index.getIndex()));
  }
  const T &operator[](Index index) const {
    assert(0 <= index.getBlock() && index.getBlock() < getBlockCount());
    getValue(index.getBlock())->getValue(index.getIndex());
    return(getValue(index.getBlock())->getValue(index.getIndex()));
  }

  T &operator[](int i) {
	  int b = 0;
	  while (b < getBlockCount() && getBlock(b)->getCount() <= i) {
		  i -= getBlock(b++)->getCount();
	  }
	  return(getBlock(b)->getValue(i));
  }

  const T &operator[](int i) const {
	  int b = 0;
	  while (b < getBlockCount() && getBlock(b)->getCount() <= i) {
		  i -= getBlock(b++)->getCount();
	  }
	  return(getBlock(b)->getValue(i));
  }

  typedef typename std::vector<Block<T> *>::iterator iterator;
  iterator beginBlock() {
    return(fieldBlocks.begin());
  }
  iterator endBlock() {
    return(fieldBlocks.end());
  }
 private:
  BlockVector<T> &operator=(const BlockVector<T> &);
};

#endif /* BLOCKVECTOR_H_ */
