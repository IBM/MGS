#ifndef __C_X1_h
#define __C_X1_h

#include <memory>

class MdlContext;

class C_X1 {

   public:
      void execute(MdlContext* context);
      C_X1();
      C_X1(C_X1* rv);
      virtual void duplicate(std::auto_ptr<C_X1>& rv);
      virtual ~C_X1();

};


#endif // __C_X1_h
