#include "C_X1.h"
#include "MdlContext.h"
#include <memory>

void C_X1::execute(MdlContext* context) {

}

C_X1::C_X1() {

}

C_X1::C_X1(C_X1* rv) {

}

void C_X1::duplicate(std::auto_ptr<C_X1>& rv) {
   rv.reset(new C_X1(this));
}

C_X1::~C_X1() {

}


