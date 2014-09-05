SRCS := AndOp \
AllValidOp \
ArrayType \
Attribute \
BaseClass \
BoolType \
BValidOp \
C_array \
C_argumentToMemberMapper \
C_connection \
C_constant \
C_compCategoryBase \
C_connectionCCBase \
C_dataType \
C_dataTypeList \
C_edge \
C_edgeConnection \
C_execute \
C_finalPhase \
C_functor \
C_general \
C_generalList \
C_identifierList \
C_inAttrPSet \
C_initialize \
C_initPhase \
C_interface \
C_interfaceImplementorBase \
C_interfaceMapping \
C_interfacePointer \
C_interfacePointerList \
C_interfaceToInstance \
C_interfaceToShared \
C_instanceMapping \
C_loadPhase \
C_nameComment \
C_nameCommentList \
C_node \
C_noop \
C_outAttrPSet \
C_phase \
C_phaseIdentifier \
C_phaseIdentifierList \
C_predicateFunction \
C_production \
C_psetMapping \
C_psetToInstance \
C_psetToShared \
C_regularConnection \
C_returnType \
C_runtimePhase \
C_shared \
C_sharedMapping \
C_sharedCCBase \
C_struct \
C_toolBase \
C_triggeredFunction \
C_triggeredFunctionInstance \
C_triggeredFunctionShared \
C_typeClassifier \
C_typeCore \
C_userFunction \
C_userFunctionCall \
C_variable \
C_computeTime \
CharType \
Class \
CommandLine \
CompCategoryBase \
ComputeTimeType \
ComputeTime \
Connection \
ConnectionCCBase \
ConnectionException \
Constant \
ConstructorMethod \
CopyConstructorMethod \
CustomAttribute \
DataType \
DataTypeAttribute \
DefaultConstructorMethod \
DoubleType \
DuplicateException \
Edge \
EdgeConnection \
EdgeType \
EdgeTypeType \
EdgeSetType \
EqualOp \
FinalPhase \
FloatType \
FunctionPredicate \
Functor \
FunctorType \
FriendDeclaration \
GeneralException \
Generatable \
GreaterOp \
GreaterEqualOp \
GridType \
GSValidOp \
IncludeClass \
IncludeHeader \
InFixOp \
Initializer \
InitPhase \
InstancePredicate \
Interface \
InterfaceImplementorBase \
InterfaceMapping \
InterfaceMappingElement \
InterfaceToMember \
InternalException \
IntType \
LensType \
LessOp \
LessEqualOp \
LoadPhase \
LongDoubleType \
LongType \
MacroConditional \
MdlContext \
MdlLexer \
Method \
MemberToInterface \
NameComment \
NDPairListType \
Node \
NodeType \
NodeTypeType \
NodeSetType \
NotEqualOp \
NotFoundException \
Operation \
Option \
OrOp \
ParanthesisOp \
Parser \
Phase \
PhaseType \
PhaseTypeInstance \
PhaseTypeShared \
PhaseTypeGridLayers \
Predicate \
PredicateException \
PSetPredicate \
PSetToMember \
RegularConnection \
RepertoireType \
ParameterSetType \
PredicateFunction \
RuntimePhase \
ServiceType \
SharedCCBase \
SharedPredicate \
ShortType \
SignedType \
StringType \
StructType \
SyntaxErrorException \
TerminalOp \
ToolBase \
TriggerType \
TriggeredFunction \
TriggeredFunctionInstance \
TriggeredFunctionShared \
TypeDefinition \
UnsignedType \
UserFunction \
UserFunctionCall \
Utility \
Variable \
VoidType \
# C_argumentMapping \


OBJS += $(patsubst %,obj/%.o,$(SRCS))
