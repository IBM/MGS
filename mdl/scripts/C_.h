// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef __C_X1_h
#define __C_X1_h

#include <memory>

class MdlContext;

class C_X1 {

   public:
      void execute(MdlContext* context);
      C_X1();
      C_X1(C_X1* rv);
      virtual void duplicate(std::unique_ptr<C_X1>&& rv);
      virtual ~C_X1();

};


#endif // __C_X1_h
