%{
#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "LensLexer.h"
#include "LensParser.h"
#include "LensParserClasses.h"
#include "LensContext.h"
#include "SimInitializer.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"

#ifndef LINUX
#include <fptrap.h>
#endif

#include <iostream>
#ifndef DISABLE_PTHREADS
#include <errno.h>
#endif
#include <pthread.h>
#include <string>
using namespace std;

#define USABLE
#define YYPARSE_PARAM parm
#define YYLEX_PARAM parm
#define YYDEBUG 1
#define CONTEXT ((LensContext *) parm)
#define yyparse lensparse
#define yyerror lenserror
#define yylex   lenslex
#define CURRENTFILE (((LensContext *) parm)->lexer)->currentFileName
#define CURRENTLINE (((LensContext *) parm)->lexer)->lineCount

// Error macros
#define PRARGMISMATCH "possible paranthesis mismatch, use ( Argument )"
#define PRARLMISMATCH "possible paranthesis mismatch, use ( ArgumentList )"
#define PRPTLMISMATCH "possible paranthesis mismatch, use ( ParameterTypeList )"
#define EXPDECL "expecting Declarator"
#define PRCRLMISMATCH "possible curly brackets mismatch use Declarator { GridDefinitionBody } or Declarator { GridDefinitionBody } Declarator"
#define EXPDECLBEFCURL "expecting a declarator before {"
   
   void lenserror(char *s);
   int lenslex(YYSTYPE *lvalp, YYLTYPE *locp, void *context);
   inline void HIGH_LEVEL_EXECUTE(void* parm, C_production* l) {
	 l->checkChildren();
	 if (l->isError()) {
	    l->recursivePrint();
	    cerr << endl; 
	    CONTEXT->setError();
	 } else {
	    CONTEXT->addStatement(l);
	 }
   }

   // composite_statement_list, grid_translation_unit, and connection_script_definition_body
   // shows the line numbers beginning of the function (Composite, Grid, 
   // or ConnectionScript) that they belong to instead of their own. Also they prsent
   // themselves as the real functions rather than  function bodies. 
   // This is done for Runtime SyntaxError Reporting.
   // -----
   // The types of which the error messages are reported no matter what are:
   // Children of Directive
   // Declaration's
   // composite_statement_list
   // grid_translation_unit
   // connection_script_definition_body
   // In other words, anything with a ';'

%}

%pure_parser
%locations

%{
#ifndef YYSTYPE_DEFINITION
#define YYSTYPE_DEFINITION
%}


%union {
      double V_double;
      int V_int;
      std::string *P_string;
      C_directive *P_directive;
      C_definition *P_definition;
      C_type_definition *P_type_definition;
      C_functor_definition *P_functor_definition;
      C_functor_specifier *P_functor_specifier;
      C_functor_category *P_functor_category;
      C_declaration *P_declaration;
      C_repertoire_declaration *P_repertoire_declaration;
      C_type_specifier *P_type_specifier;
      C_initializable_type_specifier *P_initializable_type_specifier;
      C_non_initializable_type_specifier *P_non_initializable_type_specifier;
      C_stride_list *P_stride_list;
      C_steps *P_steps;
      C_stride *P_stride;
      C_order *P_order;
      C_matrix_type_specifier *P_matrix_type_specifier;
      C_matrix_init_declarator *P_matrix_init_declarator;
      C_declarator *P_declarator;
      C_parameter_type_list *P_parameter_type_list;
      C_parameter_type *P_parameter_type;
      C_parameter_type_pair *P_parameter_type_pair;
      C_init_attr_type_node *P_init_attr_type_node;
      C_init_attr_type_edge *P_init_attr_type_edge;
      C_matrix_initializer *P_matrix_initializer;
      C_matrix_initializer_list *P_matrix_initializer_list;
      C_default_clause *P_default_clause;
      C_matrix_initializer_clause_list *P_matrix_initializer_clause_list;
      C_matrix_initializer_clause *P_matrix_initializer_clause;
      C_matrix_initializer_expression *P_matrix_initializer_expression;
      C_int_constant_list *P_int_constant_list;
      C_constant_list *P_constant_list;
      C_ndpair_clause_list *P_ndpair_clause_list;
      C_ndpair_clause_list_body *P_ndpair_clause_list_body;
      C_ndpair_clause *P_ndpair_clause;
      C_name *P_name;
  //C_value *P_value;
      C_argument_list *P_argument_list;
      C_argument *P_argument;
      C_logical_NOT_expression *P_logical_NOT_expression;
      C_logical_OR_expression *P_logical_OR_expression;
      C_logical_AND_expression *P_logical_AND_expression;
      C_equality_expression *P_equality_expression;
      C_primary_expression *P_primary_expression;
      C_gridset *P_gridset;
      C_repname *P_repname;
      C_preamble *P_preamble;
      C_gridnodeset *P_gridnodeset;
      C_nodeset *P_nodeset;
      C_nodeset_extension *P_nodeset_extension;
      C_declarator_nodeset_extension *P_declarator_nodeset_extension;
      C_relative_nodeset *P_relative_nodeset;
      C_node_type_set_specifier *P_node_type_set_specifier;
      C_node_type_set_specifier_clause *P_node_type_set_specifier_clause;
      C_layer_set *P_layer_set;
      C_layer_entry *P_layer_entry;
      C_name_range *P_name_range;
      C_layer_name *P_layer_name;
      C_edgeset *P_edgeset;
      C_edgeset_extension *P_edgeset_extension;
      C_index_set_specifier *P_index_set_specifier;
      C_index_set *P_index_set;
      C_index_entry *P_index_entry;
      C_grid_definition_body *P_grid_definition_body;
      C_dim_declaration *P_dim_declaration;
      C_grid_translation_unit *P_grid_translation_unit;
      C_grid_translation_declaration_list *P_grid_translation_declaration_list;
      C_grid_translation_declaration *P_grid_translation_declaration;
      C_grid_function_specifier *P_grid_function_specifier;
      C_grid_function_name *P_grid_function_name;
      C_composite_definition_body *P_composite_definition_body;
      C_composite_statement_list *P_composite_statement_list;
      C_composite_statement *P_composite_statement;
      C_complex_functor_definition *P_complex_functor_definition;
      C_complex_functor_declaration_body *P_complex_functor_declaration_body;
      C_complex_functor_clause_list *P_complex_functor_clause_list;
      C_complex_functor_clause *P_complex_functor_clause;
      C_constructor_clause *P_constructor_clause;
      C_function_clause *P_function_clause;
      C_return_clause *P_return_clause;
      C_connection_script_definition *P_connection_script_definition;
      C_connection_script_definition_body *P_connection_script_definition_body;
      C_connection_script_declaration *P_connection_script_declaration;
      C_constant *P_constant;
      C_trigger *P_trigger;
      C_trigger_specifier *P_trigger_specifier;
      C_service *P_service;
      C_query *P_query;
      C_query_field_entry *P_query_field_entry;
      C_query_field_set *P_query_field_set;
      C_query_field_entry_list *P_query_field_entry_list;
      C_query_path *P_query_path;
      C_query_list *P_query_list;
      C_query_path_product *P_query_path_product;
      C_system_call *P_system_call;
      C_phase *P_phase;
      C_phase_list *P_phase_list;
      C_phase_mapping *P_phase_mapping;
      C_phase_mapping_list *P_phase_mapping_list;
      C_separation_constraint *P_separation_constraint;
      C_separation_constraint_list *P_separation_constraint_list;
      std::string *P_string_literal_list;
} 

%{
#endif
%}

%token TOOLTYPE
%token <P_string> IDENTIFIER
%token <V_int> INT_CONSTANT
%token STRING
%token <P_string> STRING_LITERAL
%token <V_double>  FLOAT_CONSTANT
%token SAMPFCTR1
%token SAMPFCTR2
%token SUBNODESETFCTR
%token LAYOUT
%token NODEINITIALIZER
%token NODETYPESET
%token INITNODES
%token EDGEINITIALIZER
%token INATTRINITIALIZER
%token NDPAIRLISTFUNCTOR
%token LOCAL_COMPLEX // The use of LOCAL is to avoid conflicts with compilation conflicts with mpi.h
%token <V_int> EXP_OR
%token EXP_XOR
%token EXP_AND
%token EQUIVALENT
%token NOT_EQUIVALENT
%token MEMBER
%token TYPEDEF
%token INITIALIZE
%token FUNCTION
%token FUNCTOR
%token DIMENSION
%token TYPE
%token NODEINDEX
%token EDGEINDEX
%token DEFAULT
%token NODEINIT
%token OUT
%token EDGEINIT
%token IN
%token LOCAL_INT // The use of LOCAL is to avoid conflicts with compilation conflicts with mpi.h
%token LOCAL_FLOAT // The use of LOCAL is to avoid conflicts with compilation conflicts with mpi.h
%token MATRIX
%token NDPAIR
%token NDPAIRLIST
%token LIST
%token PSET
%token GRID
%token COMPOSITE
%token CONNECTOR
%token CONNECTIONSCRIPT
%token GRIDCOORD
%token REPNAME
%token NODESET
%token EDGESET
%token CONSTANTTYPE
%token VARIABLETYPE
%token NODETYPE
%token EDGETYPE
%token SLASH
%token DOT
%token STRIDE
%token LAYER
%token RETURN
%token INDEXSET
%token RELNODESET
%token REFPTGEN
%token MINUS
%token PLUS
%token ELLIPSIS 
%token TRIGGER
%token STRUCT
%token PUBLISHER
%token SERVICE
%token ARROW
%token COLON
%token DOUBLE_COLON
%token TRIPLE_COLON
%token ON
%token STOP
%token PAUSE
%token PORT
%token SYSTEM
%token INITPHASES
%token RUNTIMEPHASES
%token LOADPHASES
%token FINALPHASES
%token SEPARATIONCONSTRAINT
%token GRANULEMAPPER

/* types for non-terminals */
%type <P_directive> directive
%type <P_definition> definition
%type <P_type_definition> type_definition
%type <P_functor_definition> functor_definition
%type <P_functor_specifier> functor_specifier
%type <P_functor_category> functor_category
%type <P_declaration> declaration
%type <P_repertoire_declaration> repertoire_declaration
%type <P_type_specifier> type_specifier
%type <P_initializable_type_specifier> initializable_type_specifier
%type <P_non_initializable_type_specifier> non_initializable_type_specifier
%type <P_stride_list> stride_list
%type <P_steps> steps
%type <P_stride> stride
%type <P_order> order
%type <P_matrix_type_specifier> matrix_type_specifier
%type <P_matrix_init_declarator> matrix_init_declarator
%type <P_declarator> declarator
%type <P_parameter_type_list> parameter_type_list
%type <P_parameter_type> parameter_type
%type <P_parameter_type_pair> parameter_type_pair
%type <P_init_attr_type_node> init_attr_type_node
%type <P_init_attr_type_edge> init_attr_type_edge
%type <P_matrix_initializer> matrix_initializer
%type <P_matrix_initializer_list> matrix_initializer_list
%type <P_default_clause> default_clause
%type <P_matrix_initializer_clause_list> matrix_initializer_clause_list
%type <P_matrix_initializer_clause> matrix_initializer_clause
%type <P_matrix_initializer_expression> matrix_initializer_expression
%type <P_int_constant_list> int_constant_list
%type <P_constant_list> constant_list
%type <P_ndpair_clause_list> ndpair_clause_list
%type <P_ndpair_clause_list_body> ndpair_clause_list_body
%type <P_ndpair_clause> ndpair_clause
%type <P_name> name
//%type <P_value> value
%type <P_argument_list> argument_list
%type <P_argument> argument
%type <P_argument> nonempty_argument
%type <P_logical_NOT_expression> logical_NOT_expression
%type <P_logical_OR_expression> logical_OR_expression
%type <P_logical_AND_expression> logical_AND_expression
%type <P_equality_expression> equality_expression
%type <P_primary_expression> primary_expression
%type <P_gridset> gridset
%type <P_repname> repname
%type <P_preamble> preamble
%type <P_gridnodeset> gridnodeset
%type <P_nodeset> nodeset
%type <P_nodeset_extension> nodeset_extension
%type <P_declarator_nodeset_extension> declarator_nodeset_extension
%type <P_relative_nodeset> relative_nodeset
%type <P_node_type_set_specifier> node_type_set_specifier
%type <P_node_type_set_specifier_clause> node_type_set_specifier_clause
%type <P_layer_set> layer_set
%type <P_layer_entry> layer_entry
%type <P_name_range> name_range
%type <P_layer_name> layer_name
%type <P_edgeset> edgeset
%type <P_edgeset_extension> edgeset_extension
%type <P_index_set_specifier> node_index_set_specifier
%type <P_index_set_specifier> edge_index_set_specifier
%type <P_index_set> index_set
%type <P_index_entry> index_entry
%type <P_grid_definition_body> grid_definition_body
%type <P_dim_declaration> dim_declaration
%type <P_grid_translation_unit> grid_translation_unit
%type <P_grid_translation_declaration_list> grid_translation_declaration_list
%type <P_grid_translation_declaration> grid_translation_declaration
%type <P_grid_function_specifier> grid_function_specifier
%type <P_grid_function_name> grid_function_name
%type <P_composite_definition_body> composite_definition_body
%type <P_composite_statement_list> composite_statement_list
%type <P_composite_statement> composite_statement
%type <P_complex_functor_definition> complex_functor_definition
%type <P_complex_functor_declaration_body> complex_functor_declaration_body
%type <P_complex_functor_clause_list> complex_functor_clause_list
%type <P_complex_functor_clause> complex_functor_clause
%type <P_constructor_clause> constructor_clause
%type <P_function_clause> function_clause
%type <P_return_clause> return_clause
%type <P_connection_script_definition> connection_script_definition
%type <P_connection_script_definition_body> connection_script_definition_body
%type <P_connection_script_declaration> connection_script_declaration
%type <P_constant> constant
%type <P_string_literal_list> string_literal_list
%type <P_trigger> trigger;
%type <P_trigger_specifier> trigger_specifier;
%type <P_service> service;
%type <P_query> query;
%type <P_query_field_entry> query_field_entry;
%type <P_query_field_set> query_field_set;
%type <P_query_field_entry_list> query_field_entry_list;
%type <P_query_list> query_list;
%type <P_query_path> query_path;
%type <P_query_path_product> query_path_product;
%type <P_system_call> system_call;
%type <P_phase> phase;
%type <P_phase_list> phase_list;
%type <P_phase_mapping> phase_mapping;
%type <P_phase_mapping_list> phase_mapping_list;
%type <P_separation_constraint> separation_constraint;
%type <P_separation_constraint_list> separation_constraint_list;

%left EXP_AND EXP_OR EXP_XOR

%start spec_file

%%

spec_file: parser_line_list {

}
;

parser_line_list: parser_line {

}
| parser_line_list parser_line {

}
;

parser_line: definition {
   HIGH_LEVEL_EXECUTE(parm, $1);
}
| declaration {
   HIGH_LEVEL_EXECUTE(parm, $1);
}
| directive {
   HIGH_LEVEL_EXECUTE(parm, $1);
}
| error ';' {
   LensLexer *l = ((LensContext *) parm)->lexer;

   cerr << "Error at file:" << l->currentFileName << ", line:" 
	<< l->lineCount<< ", " << "unexpected token: " 
	<< l->getToken() << endl << endl;

   CONTEXT->setError();
}
| error {
   LensLexer *l = ((LensContext *) parm)->lexer;

   cerr << "Error at file:" << l->currentFileName << ", line:" 
	<< l->lineCount<< ", " << "unexpected token: " 
	<< l->getToken() << endl << endl;

   CONTEXT->setError();
}
;

error_list: error {
   if (CONTEXT->isError() == false) {
      cerr << "Error at file:" << CONTEXT->lexer->currentFileName << ", line:" 
	   << CONTEXT->lexer->lineCount<< ", " << "unexpected token: " 
	   << CONTEXT->lexer->getToken() << endl << endl;
   }
   CONTEXT->setError();
}
| error_list error {}
;

string_literal_list: STRING_LITERAL {
  $$ = $1;
}
| string_literal_list STRING_LITERAL {
  $$ = new std::string(*$1+*$2);
}

/* Directives */

directive: functor_specifier {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Directive", "FunctorSpecifier");
   $$ = new C_directive($1, localError);
}
| trigger_specifier {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Directive", "TriggerSpecifier");
   $$ = new C_directive($1, localError);
}
| system_call {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Directive", "SystemCall");
   $$ = new C_directive($1, localError);
}
;

/* Definitions */
definition: functor_definition {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Definition", "FunctorDefinition");
   $$ = new C_definition_functor($1, localError);
}
| type_definition {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Definition", "TypeDefinition");
   $$ = new C_definition_type($1, localError);
}
| CONSTANTTYPE declarator ';' {
   std::string mes = "ConstantType \"";
   mes += $2->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_definition_constanttype($2, localError);
}
| CONSTANTTYPE error_list ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Definition or Declaration", "ConstantType", EXPDECL, 
		      true);
   localError->setOriginal();
   $$ = new C_definition_constanttype(0, localError);
}
| VARIABLETYPE declarator ';' {
   std::string mes = "VariableType \"";
   mes += $2->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_definition_variabletype($2, 0, localError);
}
| VARIABLETYPE declarator '{' phase_mapping_list '}' ';' {
   std::string mes = "VariableType \"";
   mes += $2->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_definition_variabletype($2, $4, localError);
}
| VARIABLETYPE error_list ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, "Definition or Declaration", 
		      "VariableType", EXPDECL, true);
   localError->setOriginal();
   $$ = new C_definition_variabletype(0, 0, localError);
}
| NODETYPE declarator ';' {
   std::string mes = "NodeType \"";
   mes += $2->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_definition_nodetype($2, 0, 0, localError);
}
| NODETYPE declarator '{' phase_mapping_list '}' ';' {
   std::string mes = "NodeType \"";
   mes += $2->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_definition_nodetype($2, 0, $4, localError);
}
| NODETYPE declarator '(' argument ')' ';' {
   std::string mes = "NodeType \"";
   mes += $2->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_definition_nodetype($2, $4, 0, localError);
}
| NODETYPE declarator '(' argument ')' '{' phase_mapping_list '}' ';' {
   std::string mes = "NodeType \"";
   mes += $2->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_definition_nodetype($2, $4, $7, localError);
}
| NODETYPE error_list ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, "Definition", "NodeType", 
		      EXPDECL, true);
   localError->setOriginal();
   $$ = new C_definition_nodetype(0, 0, 0, localError);
}
| EDGETYPE declarator ';' {
   std::string mes = "EdgeType \"";
   mes += $2->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_definition_edgetype($2, 0, 0, localError);
}
| EDGETYPE declarator '{' phase_mapping_list '}' ';' {
   std::string mes = "EdgeType \"";
   mes += $2->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_definition_edgetype($2, 0, $4, localError);
}
| EDGETYPE declarator '(' argument ')' ';' {
   std::string mes = "EdgeType \"";
   mes += $2->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_definition_edgetype($2, $4, 0, localError);
}
| EDGETYPE declarator '(' argument ')' '{' phase_mapping_list '}' ';' {
   std::string mes = "EdgeType \"";
   mes += $2->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_definition_edgetype($2, $4, $7, localError);
}
| EDGETYPE error_list ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, "Definition", "EdgeType", 
		      EXPDECL, true);
   localError->setOriginal();
   $$ = new C_definition_edgetype(0, 0, 0, localError);
}
| TRIGGER declarator '(' parameter_type_list ')' ';' {
   std::string mes = "Trigger \"";
   mes += $2->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_definition_trigger($2, $4, localError);
}
| TRIGGER error_list ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Definition or Declaration", "Trigger", EXPDECL, true);
   localError->setOriginal();
   $$ = new C_definition_trigger(0, 0, localError);
}
| GRANULEMAPPER declarator '(' parameter_type_list ')' ';' {
   std::string mes = "GranuleMapper \"";
   mes += $2->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_definition_grid_granule($2, $4, localError);
}
| GRANULEMAPPER error_list ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Definition or Declaration", "GranuleMapper", EXPDECL, true);
   localError->setOriginal();
   $$ = new C_definition_grid_granule(0, 0, localError);
}
| STRUCT declarator ';' {
   std::string mes = "Struct \"";
   mes += $2->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_definition_struct($2, localError);
}
| STRUCT error_list ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Definition or Declaration", "Struct", EXPDECL, true);
   localError->setOriginal();
   $$ = new C_definition_struct(0, localError);
}
;

type_definition: GRID declarator '{' grid_definition_body '}' ';' {
   std::string mes = "Grid \"";
   mes += $2->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   $4->setTdError(localError->duplicate());
   localError->setOriginal();
   $$ = new C_type_definition($2, $4, localError);
}
| GRID declarator '{' grid_definition_body '}' declarator ';' {
   std::string mes = "Grid \"";
   mes += $2->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   $4->setTdError(localError->duplicate());
   localError->setOriginal();
   $$ = new C_type_definition($2, $4, $6, localError);
}
| GRID error_list ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "TypeDefinition", "Grid ", PRCRLMISMATCH, true);
   localError->setOriginal();
   $$ = new C_type_definition(localError);
}
| GRID error_list '{' grid_definition_body '}' ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, "TypeDefinition", 
		      "Grid Declarator { GridDefinitionBody }", 
		      EXPDECLBEFCURL, true);
   localError->setOriginal();
   $$ = new C_type_definition(0, $4, localError);
}
| GRID error_list '{' grid_definition_body '}' declarator ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, "TypeDefinition", 
		      "Grid Declarator { GridDefinitionBody } Declarator", 
		      EXPDECLBEFCURL, true);
   localError->setOriginal();
   $$ = new C_type_definition(0, $4, $6, localError);
}
| COMPOSITE  declarator '{' composite_definition_body '}' ';' {
   std::string mes = "Composite \"";
   mes += $2->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   $4->setTdError(localError->duplicate());
   localError->setOriginal();
   $$ = new C_type_definition($2, $4, localError);
}
| COMPOSITE  declarator '{' composite_definition_body '}' declarator ';' {
   std::string mes = "Composite \"";
   mes += $2->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   $4->setTdError(localError->duplicate());
   localError->setOriginal();
   $$ = new C_type_definition($2, $4, $6, localError);
}
| COMPOSITE error_list ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, "TypeDefinition", 
		      "Composite Declarator { CompositeDefinitionBody }", 
		      PRCRLMISMATCH, true);
   localError->setOriginal();
   $$ = new C_type_definition(localError);
}
| COMPOSITE error_list '{' composite_definition_body '}' ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, "TypeDefinition", 
		      "Composite Declarator { CompositeDefinitionBody }", 
		      EXPDECLBEFCURL, true);
   localError->setOriginal();
   $$ = new C_type_definition(0, $4, localError);
}
| COMPOSITE error_list '{' composite_definition_body '}' declarator ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, "TypeDefinition", 
		      "Composite  Declarator { CompositeDefinitionBody } Declarator", 
		      EXPDECLBEFCURL, true);
   localError->setOriginal();
   $$ = new C_type_definition(0, $4, $6, localError);
}
;

functor_definition: functor_category declarator ';' {
   std::string mes =  $1->getCategory() + " \"";
   mes += $2->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_functor_definition($1, $2, localError);
}
| functor_category declarator '(' parameter_type_list ')' ';' {
   std::string mes =  $1->getCategory() + " \"";
   mes += $2->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_functor_definition($1, $2, $4, localError);
}
| functor_category error_list ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, "Functor Definition", 
		      $1->getCategory().c_str(), EXPDECL, true);
   localError->setOriginal();
   $$ = new C_functor_definition($1, 0,  localError);
}
| functor_category declarator error_list ';' {
   std::string mes =  $1->getCategory() + " \"";
   mes += $2->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      mes.c_str(), "", PRPTLMISMATCH, true);
   localError->setOriginal();
   $$ = new C_functor_definition($1, $2, 0, localError);
}
| complex_functor_definition ';' {
   std::string mes =  "ComplexFunctorDefinition \"";
   mes += $1->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_functor_definition($1, localError);
}
| connection_script_definition ';' {
   std::string mes =  "ConnectionScriptDefinition \"";
   mes += $1->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_functor_definition($1, localError);
}
;

/* Function or declaration */


functor_specifier: declarator '(' argument_list ')' ';'{
   std::string mes =  "Functor \"";
   mes += $1->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_functor_specifier($1, $3, localError);
}
| declarator error_list ';' {
   std::string mes =  "Functor \"";
   mes += $1->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      mes.c_str(), "", PRARLMISMATCH, true);
   localError->setOriginal();
   $$ = new C_functor_specifier($1, 0, localError);
}
;

/* BINDBACK, BINDFRONT, TRAVERSE, COMPOSE *
 * should be provided outside             *
 * the grammar                            */

functor_category: CONNECTOR {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "FunctorCategory", "CONNECTOR");
   $$ = new C_functor_category("CONNECTOR", localError);
}
| SAMPFCTR1 {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "FunctorCategory", "SAMPFCTR1");
   $$ = new C_functor_category("SAMPFCTR1", localError);
}
| SAMPFCTR2 {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "FunctorCategory", "SAMPFCTR2");
   $$ = new C_functor_category("SAMPFCTR2", localError);
}
| SUBNODESETFCTR {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "FunctorCategory", "SUBNODESETFCTR");
   $$ = new C_functor_category("SUBNODESETFCTR", localError);
}
| FUNCTOR {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "FunctorCategory", "FUNCTOR");
   $$ = new C_functor_category("FUNCTOR", localError);
}
| LAYOUT {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "FunctorCategory", "LAYOUT");
   $$ = new C_functor_category("LAYOUT", localError);
}
| NODEINITIALIZER {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "FunctorCategory", "NODEINITIALIZER");
   $$ = new C_functor_category("NODEINITIALIZER", localError);
}
| EDGEINITIALIZER {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "FunctorCategory", "EDGEINITIALIZER");
   $$ = new C_functor_category("EDGEINITIALIZER", localError);
}
| INATTRINITIALIZER {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "FunctorCategory", "INATTRINITIALIZER");
   $$ = new C_functor_category("INATTRINITIALIZER", localError);
}
| NDPAIRLISTFUNCTOR {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "FunctorCategory", "NDPAIRLISTFUNCTOR");
   $$ = new C_functor_category("NDPAIRLISTFUNCTOR", localError);
}
| REFPTGEN {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "FunctorCategory", "REFPTGEN");
   $$ = new C_functor_category("REFPTGEN", localError);
}
;

/* Simple declarations */

declaration: LOCAL_INT declarator '(' INT_CONSTANT ')' ';' {
   std::string mes =  "Declaration \"";
   mes += $2->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_declaration_int($2, $4, localError);
}
| LOCAL_INT declarator '=' INT_CONSTANT ';' {
   std::string mes =  "Declaration \"";
   mes += $2->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_declaration_int($2, $4, localError);
}
| LOCAL_INT error_list ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, "Declaration", "Int", 
		      "expected Declarator = int or Declarator ( int ) ", 
		      true);
   localError->setOriginal();
   $$ = new C_declaration_int(0, 0, localError);
}
| EDGESET declarator '(' edgeset ')' ';' {
   std::string mes =  "Declaration \"";
   mes += $2->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_declaration_edgeset($2, $4, localError);
}
| EDGESET error_list ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, "Declaration", "EdseSet", 
		      "expected Declarator ( EdgeSet )", true);
   localError->setOriginal();
   $$ = new C_declaration_edgeset(0, 0, localError);
}
| TRIGGER declarator '(' trigger ')' ';' {
   std::string mes =  "Declaration \"";
   mes += $2->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_declaration_trigger($2, $4, localError);
} 
| PUBLISHER declarator '(' query_path ')' ';' {
   std::string mes =  "Declaration \"";
   mes += $2->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_declaration_publisher($2, $4, localError);
}
| PUBLISHER error_list ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Declaration", "Publisher", EXPDECL, true);
   localError->setOriginal();
   $$ = new C_declaration_publisher(0, 0, localError);
}
| PUBLISHER declarator error_list ';' {
   std::string mes =  "Declaration \"";
   mes += $2->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str(), "Publisher", 
		      "expected ( QueryPath )", true);
   localError->setOriginal();
   $$ = new C_declaration_publisher($2, 0, localError);
}
| SERVICE declarator '(' service ')' ';' {
   std::string mes =  "Declaration \"";
   mes += $2->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_declaration_service($2, $4, localError);
}
| SERVICE error_list ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Declaration", "Service", EXPDECL, true);
   localError->setOriginal();
   $$ = new C_declaration_service(0, 0, localError);
}
| SERVICE declarator error_list ';' {
   std::string mes =  "Declaration \"";
   mes += $2->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      mes.c_str(), "Service", "expected ( Service )", true);
   localError->setOriginal();
   $$ = new C_declaration_service($2, 0, localError);
}
| PORT declarator '(' INT_CONSTANT ')' ';' {
   std::string mes =  "Declaration \"";
   mes += $2->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_declaration_port($2, $4, localError);
}
| PORT error_list ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, "Declaration", "Port", 
		      "expected Declarator ( int )", true);
   localError->setOriginal();
   $$ = new C_declaration_port(0, 0, localError);
}
| LOCAL_FLOAT declarator '(' constant ')' ';' {
   std::string mes =  "Declaration \"";
   mes += $2->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_declaration_float($2, $4, localError);
}
| LOCAL_FLOAT declarator '=' constant ';' {
   std::string mes =  "Declaration \"";
   mes += $2->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_declaration_float($2, $4, localError);
}
| LOCAL_FLOAT error_list ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Declaration", "Float", "expected = float or ( float )", true);
   localError->setOriginal();
   $$ = new C_declaration_float(0, 0, localError);
}
| PSET '<' parameter_type_pair '>' declarator '(' ndpair_clause_list ')' ';' {
   std::string mes =  "Declaration \"";
   mes += $5->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_declaration_pset($3, $5, $7, localError);
}
| PSET error_list ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, "Declaration", 
		      "ParameterSet", 
		      "expected < ParameterTypePair > Declarator ( NDPairClauseList )", 
		      true);
   localError->setOriginal();
   $$ = new C_declaration_pset(0, 0, 0, localError);
}
| NDPAIR declarator '(' ndpair_clause ')' ';' {
   std::string mes =  "Declaration \"";
   mes += $2->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_declaration_ndpair($2, $4, localError);
}
| NDPAIR error_list ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, "Declaration", "NDPair", 
		      "expected Declarator ( NDPairClause )", true);
   localError->setOriginal();
   $$ = new C_declaration_ndpair(0, 0, localError);
}
| NDPAIRLIST declarator '(' ndpair_clause_list ')' ';' {
   std::string mes =  "Declaration \"";
   mes += $2->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_declaration_ndpairlist($2, $4, localError);
}
| NDPAIRLIST error_list ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, "Declaration", "NDPairList", 
		      "expected Declarator ( NDPairClauseList )", true);
   localError->setOriginal();
   $$ = new C_declaration_ndpairlist(0, 0, localError);
}
| LIST  declarator '(' '{' argument_list '}' ')' ';' {
   std::string mes =  "Declaration \"";
   mes += $2->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_declaration_list_parameter($2, $5, localError);
}
| LIST  declarator '='  '{' argument_list '}' ';' {
   std::string mes =  "Declaration \"";
   mes += $2->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_declaration_list_parameter($2, $5, localError);
}
| LIST '<' type_specifier '>' declarator '(' '{' argument_list '}' ')' ';' {
   std::string mes =  "Declaration \"";
   mes += $5->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_declaration_list_parameter($3, $5, $8, localError);
}
| LIST '<' type_specifier '>' declarator '='  '{' argument_list '}' ';' {
   std::string mes =  "Declaration \"";
   mes += $5->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_declaration_list_parameter($3, $5, $8, localError);
}
| LIST error_list ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, "Declaration", "List", 
		      "possible < > or ( )  or { } mismatch", true);
   localError->setOriginal();
   $$ = new C_declaration_list_parameter(0, 0, 0, localError);
}
| NODESET declarator '(' nodeset ')' ';' {
   std::string mes =  "Declaration \"";
   mes += $2->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_declaration_nodeset($2, $4, localError);
}
| NODESET error_list ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, "Declaration", "NodeSet", 
		      "expected  Declarator ( NodeSet )", true);
   localError->setOriginal();
   $$ = new C_declaration_nodeset(0, 0, localError);
}
| NODETYPESET declarator '(' node_type_set_specifier_clause ')'	';' {
   std::string mes =  "Declaration \"";
   mes += $2->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_declaration_node_type_set($2, $4, localError);
}
| NODETYPESET error_list ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, "Declaration", "NodeTypeSet",
		      "expected Declarator ( NodeTypeSetSpecifierClause )", 
		      true);
   localError->setOriginal();
   $$ = new C_declaration_node_type_set(0, 0, localError);
}
| RELNODESET declarator '(' relative_nodeset ')' ';' {
   std::string mes =  "Declaration \"";
   mes += $2->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_declaration_rel_nodeset($2, $4, localError);
}
| RELNODESET error_list ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, "Declaration", "RelNodeSet",
		      "expected Declarator ( RelativeNodeSet )", true);
   localError->setOriginal();
   $$ = new C_declaration_rel_nodeset(0, 0, localError);
}
| GRIDCOORD declarator '(' gridset ')' ';' {
   std::string mes =  "Declaration \"";
   mes += $2->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_declaration_gridcoord($2, $4, localError);
}
| GRIDCOORD error_list ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, "Declaration", "GridCoord",
		      "expected Declarator ( GridSet )", true);
   localError->setOriginal();
   $$ = new C_declaration_gridcoord(0, 0, localError);
}
| REPNAME declarator '(' repname ')' ';' {
   std::string mes =  "Declaration \"";
   mes += $2->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_declaration_repname($2, $4, localError);
}
| REPNAME error_list ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, "Declaration", "RepName", 
		      "expected Declarator ( RepName )", true);
   localError->setOriginal();
   $$ = new C_declaration_repname(0, 0, localError);
}
| INDEXSET declarator '(' index_set ')' ';' {
   std::string mes =  "Declaration \"";
   mes += $2->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_declaration_index_set($2, $4, localError);
}
| INDEXSET error_list ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, "Declaration", "IndexSet", 
		      "expected Declarator ( IndexSet )", true);
   localError->setOriginal();
   $$ = new C_declaration_index_set(0, 0, localError);
}
| matrix_type_specifier matrix_init_declarator ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, "Declaration", 
		      "MatrixTypeSpecifier MatrixInitDeclarator");
   localError->setOriginal();
   $$ = new C_declaration_matrix_type($1, $2, localError);
}
| matrix_type_specifier error_list ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, "Declaration", 
		      "MatrixTypeSpecifier", "expected MatrixInitDeclarator", 
		      true);
   localError->setOriginal();
   $$ = new C_declaration_matrix_type($1, 0, localError);
}
| STRIDE declarator '(' stride_list ')'	';' {
   std::string mes =  "Declaration \"";
   mes += $2->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_declaration_stride($2, $4, localError);
}
| STRIDE error_list ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, "Declaration", "Stride", 
		      "expected Declarator ( StrideList )", true);
   localError->setOriginal();
   $$ = new C_declaration_stride(0, 0, localError);
}
| declarator declarator '(' argument_list ')' ';' {
   std::string mes =  "Declaration " + $1->getName() + " \"";
   mes += $2->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_declaration_decl_decl_args($1, $2, $4, localError);
}
| declarator declarator ndpair_clause_list ';' {
   std::string mes =  "Declaration " + $1->getName() + " \"";
   mes += $2->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_declaration_decl_decl_args($3, $1, $2, localError);
}
| declarator declarator error_list  ';' {
   std::string mes =  "Declaration " + $1->getName() + " \"";
   mes += $2->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str(), "", 
		      "expected () or ( ArgumentList ) or < NDPairList >", 
		      true);
   localError->setOriginal();
   $$ = new C_declaration_decl_decl_args($1, $2, 0, localError);
}
| repertoire_declaration ';' {
   std::string mes =  "Declaration " + $1->getType() + " \"";
   mes += $1->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_declaration_repertoire($1, localError);
}
| INITPHASES '=' '{' phase_list '}' ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "InitPhases", " = { PhaseList }");
   $$ = new C_declaration_init_phases($4, localError);   
}
| RUNTIMEPHASES '=' '{' phase_list '}' ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "RuntimePhases", " = { PhaseList }");
   $$ = new C_declaration_runtime_phases($4, localError);   
}
| LOADPHASES '=' '{' phase_list '}' ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "LoadPhases", " = { PhaseList }");
   $$ = new C_declaration_load_phases($4, localError);   
}
| FINALPHASES '=' '{' phase_list '}' ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "FinalPhases", " = { PhaseList }");
   $$ = new C_declaration_final_phases($4, localError);   
}
| SEPARATIONCONSTRAINT '(' separation_constraint_list ')' ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "SeparationConstraint", 
		      " ( SeparationConstraintList )");
   $$ = new C_declaration_separation_constraint($3, localError);   
}
;

separation_constraint_list: separation_constraint {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "SeparationConstraintList", "SeparationConstraint");
   $$ = new C_separation_constraint_list($1, localError);   
}
| separation_constraint_list ',' separation_constraint {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "SeparationConstraintList", 
		      "SeparationConstraintList , SeparationConstraint");
   $$ = new C_separation_constraint_list($1, $3, localError);      
}
;

separation_constraint: declarator {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "SeparationConstraint", "Declarator");
   $$ = new C_separation_constraint($1, localError);
}
| nodeset {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "SeparationConstraint", "NodeSet");
   $$ = new C_separation_constraint($1, localError);
}
;

repertoire_declaration: declarator declarator {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "repertoire_declaration", "Declarator Declarator");
   $$ = new C_repertoire_declaration($1, $2, localError);
}
;

type_specifier: initializable_type_specifier {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "TypeSpecifier", "InitializableTypeSpecifier");
   $$ = new C_type_specifier($1, localError);
}
| non_initializable_type_specifier {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "TypeSpecifier", "NonInitializableTypeSpecifier");
   $$ = new C_type_specifier($1, localError);
}
;

initializable_type_specifier: PSET '<' parameter_type_pair '>' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, "InitializableTypeSpecifier",
		      "Pset < ParameterTypePair >");
   $$ = new C_initializable_type_specifier(C_type_specifier::_PSET, $3, 
					   localError);
}
| REPNAME {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "InitializableTypeSpecifier", "RepName");
   $$ = new C_initializable_type_specifier(C_type_specifier::_REPNAME, 
					   localError);
}
| LIST '<' type_specifier '>' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "InitializableTypeSpecifier", "List < TypeSpecifier >");
   $$ = new C_initializable_type_specifier(C_type_specifier::_LIST, $3, 
					   localError);
}
| GRIDCOORD {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "InitializableTypeSpecifier", "GridCoord");
   $$ = new C_initializable_type_specifier(C_type_specifier::_GRIDCOORD, 
					   localError);
}
| NDPAIR {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "InitializableTypeSpecifier", "NDPair");
   $$ = new C_initializable_type_specifier(C_type_specifier::_NDPAIR, 
					   localError);
}
| LOCAL_INT {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "InitializableTypeSpecifier", "int");
   $$ = new C_initializable_type_specifier(C_type_specifier::_INT, localError);
}
| LOCAL_FLOAT	{
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "InitializableTypeSpecifier", "float");
   $$ = new C_initializable_type_specifier(C_type_specifier::_FLOAT, 
					   localError);
}
| STRING {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "InitializableTypeSpecifier", "string");
   $$ = new C_initializable_type_specifier(C_type_specifier::_STRING, 
					   localError);
}
| RELNODESET {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "InitializableTypeSpecifier", "RelNodeSet");
   $$ = new C_initializable_type_specifier(C_type_specifier::_RELNODESET, 
					   localError);
}
| NODESET {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "InitializableTypeSpecifier", "NodeSet");
   $$ = new C_initializable_type_specifier(C_type_specifier::_NODESET, 
					   localError);
}
| SERVICE {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "InitializableTypeSpecifier", "Service");
   $$ = new C_initializable_type_specifier(C_type_specifier::_SERVICE, 
					   localError);
}
| EDGESET {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "InitializableTypeSpecifier", "EdgeSet");
   $$ = new C_initializable_type_specifier(C_type_specifier::_EDGESET, 
					   localError);
}
| GRANULEMAPPER {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "InitializableTypeSpecifier", "GranuleMapper");
   $$ = new C_initializable_type_specifier(C_type_specifier::_GRANULEMAPPER, 
					   localError);
}
| PORT {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "InitializableTypeSpecifier", "Port");
   $$ = new C_initializable_type_specifier(C_type_specifier::_PORT, 
					   localError);
}
| TRIGGER {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "InitializableTypeSpecifier", "Trigger");
   $$ = new C_initializable_type_specifier(C_type_specifier::_TRIGGER, 
					   localError);
}
;

non_initializable_type_specifier: PSET {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "NonInitializableTypeSpecifier", "Pset");
   $$ = new C_non_initializable_type_specifier(C_type_specifier::_PSET, 
					       localError);
}
| NODETYPE {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "NonInitializableTypeSpecifier", "NodeType");
   $$ = new C_non_initializable_type_specifier(C_type_specifier::_NODETYPE, 
					       localError);
}
| EDGETYPE {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "NonInitializableTypeSpecifier", "EdgeType");
   $$ = new C_non_initializable_type_specifier(C_type_specifier::_EDGETYPE, 
					       localError);
}
;

stride_list: '{' steps '}' '{' stride '}' '{' order '}'	{
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @2.first_line, 
		      "StrideList", "{ Steps } { Stride } { Order }");
   $$ = new C_stride_list($2, $5, $8, localError);
}
;

steps: int_constant_list {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Steps", "IntConstantList");
   $$ = new C_steps($1, localError);
}
;

stride: int_constant_list {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Stride", "IntConstantList");
   $$ = new C_stride($1, localError);
}
;

order: int_constant_list {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Order", "IntConstantList");
   $$ = new C_order($1, localError);
}
;

matrix_type_specifier: MATRIX '<' type_specifier '>' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "MatrixTypeSpecifier", "Matrix < TypeSpecifier >");
   $$ = new C_matrix_type_specifier($3, localError);
}
;

matrix_init_declarator: declarator '[' int_constant_list ']' '=' matrix_initializer {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, "MatrixInitDeclarator", 
		      "Declarator [ IntConstantList ] = MatrixInitializer");
   $$ = new C_matrix_init_declarator($1, $3, $6, localError);
}
| declarator '[' int_constant_list ']' '(' matrix_initializer ')' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, "MatrixInitDeclarator", 
		      "Declarator [ IntConstantList ] ( MatrixInitializer )");
   $$ = new C_matrix_init_declarator($1, $3, $6, localError);
}
;

declarator: IDENTIFIER {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Declarator", "Identifier");
   $$ = new C_declarator($1, localError);
} 
;

parameter_type_list: parameter_type {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "ParameterTypeList", "ParameterType");
   $$ = new C_parameter_type_list($1, localError);
}
| parameter_type_list ',' parameter_type {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, "ParameterTypeList", 
		      "ParameterTypeList , ParameterType");
   $$ = new C_parameter_type_list($1, $3, localError);
}
;

parameter_type: /* empty */ {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, CURRENTLINE, 
		      "ParameterType", "Empty");
   $$ = new C_parameter_type(localError);
}
| ELLIPSIS {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "ParameterType", "...");
   $$ = new C_parameter_type(false, localError);
}
| type_specifier {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "ParameterType", "TypeSpecifier");
   std::string* a = new std::string("");
   $$ = new C_parameter_type($1, a, localError);
}
| matrix_type_specifier	{
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "ParameterType", "MatrixTypeSpecifier");
   std::string* a = new std::string("");
   $$ = new C_parameter_type($1, a, localError);
}
| functor_category {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "ParameterType", "FunctorCategory");
   std::string* a = new std::string("");
   $$ = new C_parameter_type($1, a, localError);
}
| type_specifier IDENTIFIER {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "ParameterType", "TypeSpecifier string");
   $$ = new C_parameter_type($1, $2, localError);
}	
| matrix_type_specifier IDENTIFIER {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "ParameterType", "MatrixTypeSpecifier string");
   $$ = new C_parameter_type($1, $2, localError);
}
| functor_category IDENTIFIER {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "ParameterType", "FunctorCategory string");
   $$ = new C_parameter_type($1, $2, localError);
}
;

parameter_type_pair: declarator ',' init_attr_type_node {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "ParameterTypePair", "Declarator , InitAttrTypeNode");
   $$ = new C_parameter_type_pair($1, $3, localError);
}
| declarator ',' init_attr_type_edge {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "ParameterTypePair", "Declarator , InitAttrTypeEdge");
   $$ = new C_parameter_type_pair($1, $3, localError);
}
;

init_attr_type_node: IN	{
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "InitAttrTypeNode", "In");
   $$ = new C_init_attr_type_node(0, localError);
}
| OUT {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "InitAttrTypeNode", "Out");
   $$ = new C_init_attr_type_node(1, localError);
}
| NODEINIT {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "InitAttrTypeNode", "NodeInit");
   $$ = new C_init_attr_type_node(2, localError);
}
;

init_attr_type_edge: EDGEINIT {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "InitAttrTypeEdge", "EdgeInit");
   $$ = new C_init_attr_type_edge(localError);
}
;

/* Initializers */

matrix_initializer: '{' matrix_initializer_list '}' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "MatrixInitializer", "{ MatrixInitializerList }");
   $$ = new C_matrix_initializer($2, localError);
}
;

matrix_initializer_list:  default_clause {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "MatrixInitializerList", "DefaultClause");
   $$ = new C_matrix_initializer_list($1, localError);
}
| default_clause ';' matrix_initializer_clause_list {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, "MatrixInitializerList", 
		      "DefaultClause ; MatrixInitializerClauseList");
   $$ = new C_matrix_initializer_list($1, $3, localError);
}
| matrix_initializer_clause_list {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "MatrixInitializerList", "MatrixInitializerClauseList");
   $$ = new C_matrix_initializer_list($1, localError);
}
;

default_clause: DEFAULT '(' constant ')' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Default Clause", "Default { constant )");
   $$ = new C_default_clause($3, localError);
}
;

matrix_initializer_clause_list: matrix_initializer_clause {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "MatrixInitializerClauseList", 
		      "MatrixInitializerClause");
   $$ = new C_matrix_initializer_clause_list($1, localError);
}
| matrix_initializer_clause_list matrix_initializer_clause {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "MatrixInitializerClauseList", 
		      "MatrixInitializerClauseList MatrixInitializerClause");
   $$ = new C_matrix_initializer_clause_list($1, $2, localError);
}
;

matrix_initializer_clause: constant_list ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "MatrixInitializerClause", "ConstantList ;");
   $$ = new C_matrix_initializer_clause($1, localError);
}
| matrix_initializer_expression constant_list ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "MatrixInitializerClause", 
		      "MatrixInitializerExpression ConstantList ;");
   $$ = new C_matrix_initializer_clause($1, $2, localError);
}
;

matrix_initializer_expression: '[' int_constant_list ']' '=' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @2.first_line, 
		      "MatrixInitializerExpression", "[ IntConstantList ] =");
   $$ = new C_matrix_initializer_expression($2, localError);
}
;

int_constant_list: INT_CONSTANT {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "IntConstantList", "int");
   $$ = new C_int_constant_list($1, localError);
}
| int_constant_list ',' INT_CONSTANT {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "IntConstantList", "int , int");
   $$ = new C_int_constant_list($1, $3, localError);
}
;

constant_list: constant {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Constant List", "Constant");
   $$ = new C_constant_list($1, localError);
}
| constant_list ',' constant {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Constant List", "Constant List");
   $$ = new C_constant_list($1, $3, localError);
}
;

ndpair_clause_list: '<' ndpair_clause_list_body '>' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @2.first_line, 
		      "NDPairClauseList", "< NDPairClauseListBody >");
   $$ = new C_ndpair_clause_list($2, localError);
}
| ndpair_clause_list PLUS '<' ndpair_clause_list_body '>' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, "NDPairClauseList", 
		      "NDPairClauseList + < NDPairClauseListBody >");
   $$ = new C_ndpair_clause_list($1, $4, localError);
}
| '<' '>' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, CURRENTLINE, 
		      "NDPairClauseList", "< >");
   $$ = new C_ndpair_clause_list(localError);
}
;

ndpair_clause_list_body: ndpair_clause {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "NDPairClauseListBody", "NDPairClause");
   $$ = new C_ndpair_clause_list_body($1, localError);
}
| ndpair_clause ',' ndpair_clause_list_body {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, "NDPairClauseListBody", 
		      "NDPairClause , NDPairClauseListBody");
   $$ = new C_ndpair_clause_list_body($1, $3, localError);
}
| ndpair_clause ndpair_clause_list_body {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "NDPairClauseListBody", "", "missing comma", true);
   $$ = new C_ndpair_clause_list_body($1, $2, localError);
}
;

ndpair_clause: name '=' nonempty_argument {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "NDPairClause", "Name = argument");
   $$ = new C_ndpair_clause($1, $3 , localError);
}
;

name: IDENTIFIER {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Name", "string");
   $$ = new C_name($1, localError);
}
;

/*
value: STRING_LITERAL {
  SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Value", "StringLiteral");
   $$ = new C_value($1, localError);
}
;
*/

/* Argument lists */

argument_list: argument	{
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Argument List", "Argument");
   $$ = new C_argument_list($1, localError);
}
| argument_list ',' argument {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Argument List", "Argument , Argument");
   $$ = new C_argument_list($1, $3, localError);
}
;

argument: /* empty */ {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, CURRENTLINE, 
		      "Argument", "Empty");
   $$ = new C_argument_void(localError);
}
| nonempty_argument {
   $$ = $1;
}
;

nonempty_argument: constant {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Argument", "Constant");
   $$ = new C_argument_constant($1, localError);
}
| declarator {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Argument", "Declarator");
   $$ = new C_argument_declarator($1, localError);
}
| nodeset {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Argument", "NodeSet");
   $$ = new C_argument_nodeset($1, localError);
}
| relative_nodeset {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Argument", "Relative NodeSet");
   $$ = new C_argument_relative_nodeset($1, localError);
}
| string_literal_list {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Argument", "String Literal List");
   $$ = new C_argument_string($1, localError);
}
| PSET '<' parameter_type_pair '>' '(' ndpair_clause_list ')' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Argument", "Parameter Set");
   $$ = new C_argument_pset($3, $6, localError);
}
| declarator '(' argument_list ')' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Argument", "Declarator with Argument List");
   $$ = new C_argument_decl_args($1, $3, localError);
}
| LIST '<' type_specifier '>' '(' '{' argument_list '}' ')' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, "Argument", 
		      "List and Type Specifier");
   $$ = new C_argument_argument_list($3, $7, localError);
}
| MATRIX '<' type_specifier '>' '(' '{' argument_list '}' ')' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Argument", "Matrix");
   $$ = new C_argument_matrix($3, $7, localError);
}
| '{' argument_list '}' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @2.first_line, 
		      "Argument", "Curly Brackets");
   $$ = new C_argument_argument_list($2, localError);
}
| ndpair_clause_list {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Argument", "NDPair Clause List");
   $$ = new C_argument_ndpair_clause_list($1, localError);
}
| edgeset {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Argument", "EdgeSet");
   $$ = new C_argument_edgeset($1, localError);
}
| query_path_product {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Argument", "Query Path Product");
   $$ = new C_argument_query_path_product($1, localError);
}
;


logical_NOT_expression: '!' primary_expression {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @2.first_line, 
		      "!", "!");
   $$ = new C_logical_NOT_expression($2, localError);
}
;

logical_OR_expression: logical_AND_expression {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "||", "&&");
   $$ = new C_logical_OR_expression($1, localError);
}
| logical_OR_expression EXP_OR logical_AND_expression {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "||", "|| &&");
   $$ = new C_logical_OR_expression($1, $3, localError);
}
;

logical_AND_expression: equality_expression {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "&&", "&&");
   $$ = new C_logical_AND_expression($1, localError);
}
| logical_AND_expression EXP_AND equality_expression {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "&&", "equality");
   $$ = new C_logical_AND_expression($1, $3, localError);
}
;

equality_expression: primary_expression {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Equality Expression", "Primary Expression");
   $$ = new C_equality_expression($1, localError);
}
| name EQUIVALENT string_literal_list {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Equality Expression", "= String Literal List");
   $$ = new C_equality_expression($1, $3, true, localError);
}
| name NOT_EQUIVALENT string_literal_list {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Equality Expression", "!= String Literal List");
   $$ = new C_equality_expression($1, $3, false, localError);
}
;

primary_expression: MEMBER  '(' layer_set ')' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "PrimaryExpression", "Member ( LayerSet )");
   $$ = new C_primary_expression($3, localError);
}
| '(' logical_OR_expression ')' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @2.first_line, 
		      "PrimaryExpression", "( LogicalOrExpression )");
   $$ = new C_primary_expression($2, localError);
}
| logical_NOT_expression {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "PrimaryExpression", "( LogicalOrExpression )");
   $$ = new C_primary_expression($1, localError);
}
;

/* must resolve whether the following is a gridcoord or a nodeset */

gridset: repname gridnodeset {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "GridSet", "Repname GridNodeSet");
   $$ = new C_gridset($1, $2, localError);
}
;

repname: preamble IDENTIFIER {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "RepName", "Preamble string");
   $$ = new C_repname($1, $2, localError);
}
| DOT {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "RepName", ".");
   $$ = new C_repname(localError);
}
| IDENTIFIER {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "RepName", "string");
   $$ = new C_repname($1, localError);
}
;

preamble: DOT SLASH {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Preamble", ". /");
   $$ = new C_preamble(localError);
}
| IDENTIFIER SLASH {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Preamble", "string /");
   $$ = new C_preamble($1, localError);
}
| preamble IDENTIFIER SLASH {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Preamble", "Preamble string /");
   $$ = new C_preamble($1, $2, localError);
}
;

gridnodeset: '[' index_set ']' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "GridNodeSet", "[ IndexSet ]");
   $$ = new C_gridnodeset($2, localError);
}
;

nodeset: gridset {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "NodeSet", "GridSet");
   $$ = new C_nodeset($1, localError);
}
| gridset nodeset_extension {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "NodeSet", "GridSet NodeSetExtension");
   $$ = new C_nodeset($1, $2, localError);
}
| declarator DOT declarator {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "NodeSet", "Declarator . Declarator");
   $$ = new C_nodeset($1, $3, localError);
}
| declarator_nodeset_extension {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "NodeSet", "DeclaratorNodeSetExtension");
   $$ = new C_nodeset($1, localError);
}
;

nodeset_extension: node_type_set_specifier node_index_set_specifier {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, "NodeSetExtension", 
		      "NodeTypeSetSpecifier NodeIndexSetSpecifier");
   $$ = new C_nodeset_extension($1, $2, localError);
}
| node_index_set_specifier {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, "NodeSetExtension", 
		      "NodeIndexSetSpecifier");
   $$ = new C_nodeset_extension($1, localError);
}
| node_type_set_specifier {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "NodeSetExtension", "NodeTypeSetSpecifier");
   $$ = new C_nodeset_extension($1, localError);
}
;

/* in our code we must determine whether the following is a nodeset or a
   relative_nodeset */

declarator_nodeset_extension: declarator nodeset_extension {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Declarator Nodeset Extension", "Declarator");
   $$ = new C_declarator_nodeset_extension($1, $2, localError);
}
;

relative_nodeset: gridnodeset nodeset_extension	{
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "RelativeNodeset", "GridNodeSet NodeSetExtension");
   $$ = new C_relative_nodeset($1, $2, localError);
}
| gridnodeset {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "RelativeNodeset", "GridNodeSet");
   $$ = new C_relative_nodeset($1, localError);
}
;

node_type_set_specifier: DOT LAYER '(' node_type_set_specifier_clause ')' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, "NodeTypeSetSpecifier", 
		      ". Layer ( NodeTypeSetSpecifierClause )");
   $$ = new C_node_type_set_specifier($4, localError);
}
;

node_type_set_specifier_clause: layer_set {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "NodeTypeSetSpecifierClause", "LayerSet");
   $$ = new C_node_type_set_specifier_clause($1, localError);
}
| logical_OR_expression	{
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "NodeTypeSetSpecifierClause", "LogicalOrExpression");
   $$ = new C_node_type_set_specifier_clause($1, localError);
}                       
;

layer_set: layer_entry	{
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "LayerSet", "LayerEntry");
   $$ = new C_layer_set($1, localError);
}
| layer_set ',' layer_entry {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "LayerSet", "LayerSet , LayerEntry");
   $$ = new C_layer_set($1, $3, localError);
}
;

layer_entry: layer_name	{
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "LayerEntry", "LayerName");
   $$ = new C_layer_entry($1, localError);
}
| name_range {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "LayerEntry", "NameRange");
   $$ = new C_layer_entry($1, localError);
}
;

edgeset: declarator edgeset_extension {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "EdgeSet", "Declarator Edgeset Extension");
   $$ = new C_edgeset($1, $2, localError);
}
| '[' declarator ARROW declarator ']' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @2.first_line, 
		      "EdgeSet", "[ Declarator -> Declarator ]");
   $$ = new C_edgeset($2, $4, localError);
}
| '[' declarator ARROW declarator ']' edgeset_extension {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @2.first_line, 
		      "EdgeSet", 
		      " [ Declarator -> Declarator ] Edgeset Extension");
   $$ = new C_edgeset($2, $4, $6, localError);
}
| '[' nodeset ARROW nodeset ']' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @2.first_line, 
		      "EdgeSet", "[ Nodeset -> Nodeset ]");
   $$ = new C_edgeset($2, $4, localError);
}
| '[' nodeset ARROW nodeset ']' edgeset_extension {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @2.first_line, 
		      "EdgeSet", " [ NodeSet -> NodeSet ] Edgeset Extension");
   $$ = new C_edgeset($2, $4, $6, localError);
}
| '[' declarator PLUS declarator ']' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @2.first_line, 
		      "EdgeSet", "[ Declarator + Declarator ]");
   $$ = new C_edgeset($2, $4, localError);
}
| '[' declarator PLUS declarator ']' edgeset_extension {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @2.first_line, 
		      "EdgeSet", 
		      "[ Declarator + Declarator ] Edgeset Extension");
   $$ = new C_edgeset($2, $4, $6, localError);
}
| '[' edgeset PLUS edgeset ']' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @2.first_line, 
		      "EdgeSet", "[ EdgeSet + EdgeSet ]");
   $$ = new C_edgeset($2, $4, localError);
}
| '[' edgeset PLUS edgeset ']' edgeset_extension {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @2.first_line, 
		      "EdgeSet", "[ EdgeSet + EdgeSet ] Edgeset Extension");
   $$ = new C_edgeset($2, $4, $6, localError);
}
;

edgeset_extension: DOT TYPE '(' declarator ')' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "EdgeSet Extension", ". Type ( Declarator )");
   $$ = new C_edgeset_extension($4, localError);
}
| edge_index_set_specifier {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "EdgeSet Extension", "EdgeIndexSetSpecifier");
   $$ = new C_edgeset_extension($1, localError);
}
| DOT TYPE '(' declarator ')' edge_index_set_specifier {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "EdgeSet", 
		      ". Type ( Declarator ) EdgeIndexSetSpecifier");
   $$ = new C_edgeset_extension($4, $6, localError);
}
;

name_range: layer_name MINUS  layer_name {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "NameRange", "LayerName - LayerName");
   $$ = new C_name_range($1, $3, localError);
}
;

layer_name: IDENTIFIER {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "LayerName", "string");
   $$ = new C_layer_name($1, localError);
}
;

node_index_set_specifier: DOT NODEINDEX '(' index_set ')' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "IndexSetSpecifier", ". NodeIndex ( IndexSet )");
   $$ = new C_index_set_specifier($4, localError);
}
;

edge_index_set_specifier: DOT EDGEINDEX '(' index_set ')' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "IndexSetSpecifier", ". EdgeIndex ( IndexSet )");
   $$ = new C_index_set_specifier($4, localError);
}
;

index_set: index_entry {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "IndexSet", "IndexEntry");
   $$ = new C_index_set($1, localError);
}
| index_set ',' index_entry {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "IndexSet", "IndexSet , IndexEntry");
   $$ = new C_index_set($1, $3, localError);
}
| /* empty */ {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, CURRENTLINE, 
		      "IndexSet", "Empty");
   $$ = new C_index_set(localError);
}
;

index_entry: INT_CONSTANT COLON INT_CONSTANT COLON INT_CONSTANT {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "IndexEntry", "int : int : int");
   $$ = new C_index_entry($1,$3,$5, localError);
}
| INT_CONSTANT COLON INT_CONSTANT {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "IndexEntry", "int : int");
   $$ = new C_index_entry($1,$3, localError);
}
| INT_CONSTANT {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "IndexEntry", "int");
   $$ = new C_index_entry($1, localError);
}
;

/* Repertoire Specification: Grid */

grid_definition_body: dim_declaration {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "GridDefinitionBody", "DimDeclaration");
   $$ = new C_grid_definition_body($1, localError);
}
| dim_declaration grid_translation_unit {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "GridDefinitionBody", 
		      "DimDeclaration GridTranslationUnit");
   $$ = new C_grid_definition_body($1, $2, localError);
}
| grid_translation_unit { // error condition
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "GridDefinitionBody", "", 
		      "first statement has to be a dimension declaration", 
		      true);
   $$ = new C_grid_definition_body(0, $1, localError);
}
;

dim_declaration: DIMENSION '(' int_constant_list ')' ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Dimension Declaration", "( IntList )");
   $$ = new C_dim_declaration($3, localError);
}
| DIMENSION error_list ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Dimension Declaration", "( IntList )", 
		      "possible paranthesis or comma error", true);
   localError->setOriginal();
   $$ = new C_dim_declaration(0, localError);
}

;

grid_translation_unit: grid_translation_declaration_list {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "GridTranslationUnit", "GridTranslationDeclarationList");
   $$ = new C_grid_translation_unit($1, localError);
}
;

grid_translation_declaration_list: grid_translation_declaration	{
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "GridTranslationDeclarationList", 
		      "GridTranslationDeclaration");
   $$ = new C_grid_translation_declaration_list($1, localError);
}
| grid_translation_declaration_list  grid_translation_declaration {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "GridTranslationDeclarationList", 
		      "GridTranslationDeclarationList GridTranslationDeclaration");
   $$ = new C_grid_translation_declaration_list($1,$2, localError);
}
;

grid_translation_declaration: declaration {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "GridTranslationDeclaration", "Declaration ;");
   $$ = new C_grid_translation_declaration($1, localError);
}
| grid_function_specifier {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "GridTranslationDeclaration", "GridFunctionSpecifier ;");
   $$ = new C_grid_translation_declaration($1, localError);
}
;

grid_function_specifier: directive {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "GridFunctionSpecifier", "Directive");
   $$ = new C_grid_function_specifier($1, localError);
}
| grid_function_name {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "GridFunctionSpecifier", "GridFunctionName");
   $$ = new C_grid_function_specifier($1, localError);
}
;

grid_function_name: INITNODES '(' argument_list ')' ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "GridFunctionName", "InitNodes ( ArgumentList )");
   $$ = new C_grid_function_name($3, localError);
}
| INITNODES error_list ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "GridFunctionName", "InitNodes ( ArgumentList )", 
		      "possible paranthesis or comma error", true);
   localError->setOriginal();
   $$ = new C_grid_function_name(0, localError);
}
| LAYER '(' declarator ',' argument_list ')' ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "GridFunctionName", 
		      "Layer ( Declarator , ArgumentList )");
   
   SyntaxError* error2 = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "GridFunctionName", 
		      "Layer ( Declarator , ArgumentList )", 
		      "default granule for grid");
    $$ = new C_grid_function_name($3, $5, localError);
}
| LAYER error_list ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "GridFunctionName", 
		      "Layer ( Declarator , ArgumentList )", 
		      "possible paranthesis or comma error", true);
   localError->setOriginal();
   $$ = new C_grid_function_name(0, 0, localError);
}
;

/* Repertoire Specification: Composite */


composite_definition_body: composite_statement_list {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Composite Definition Body", "Composite Statement List");
   $$ = new C_composite_definition_body($1, localError);
}
;

composite_statement_list: composite_statement {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Composite Statement List", "Composite Statement");
   $$ = new C_composite_statement_list($1, localError);
}
| composite_statement_list composite_statement {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Composite Statement List", "Composite Statement List");
   $$ = new C_composite_statement_list($1, $2, localError);
}
;

composite_statement: declaration {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Composite Statement", "Declaration");
   $$ = new C_composite_statement($1, localError);
}
| directive {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Composite Statement", "Directive");
   $$ = new C_composite_statement($1, localError);
}
;

/* Connection Specification: Connector, Connection Script */

complex_functor_definition: functor_category declarator '{' complex_functor_declaration_body '}' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Complex Functor Definition", "Functor Category");
   $$ = new C_complex_functor_definition($1, $2, $4, localError);
}
;

complex_functor_declaration_body: complex_functor_clause_list {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Complex Functor Declaration Body", 
		      "Complex Functor Clause List");
   $$ = new C_complex_functor_declaration_body($1, localError);
}
;

complex_functor_clause_list: complex_functor_clause {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Complex Functor Clause List", "Single");
   $$ = new C_complex_functor_clause_list($1, localError);
}
| complex_functor_clause_list  complex_functor_clause {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Complex Functor Clause List", "Multiple");
   $$ = new C_complex_functor_clause_list($1, $2, localError);
}
;

complex_functor_clause: constructor_clause {
   $$ = $1;
}
| function_clause {
   $$ = $1;
}
| return_clause {
   $$ = $1;
}
;     

constructor_clause: INITIALIZE '(' parameter_type_list ')' ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Constructor Clause", 
		      "Initialize ( ParameterTypeList )");
   $$ = new C_constructor_clause($3, localError);
}
| INITIALIZE error_list ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Constructor Clause", "Initialize", 
		      " expected ( ParameterTypeList )", true );
   localError->setOriginal();
   $$ = new C_constructor_clause(0, localError);
}
;

function_clause: FUNCTION '(' parameter_type_list ')' ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Function Clause", "Function ( ParameterTypeList )");
   $$ = new C_function_clause($3, localError);
}
| FUNCTION error_list ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Function Clause", "Function", 
		      " expected ( ParameterTypeList )", true);
   localError->setOriginal();
   $$ = new C_function_clause(0, localError);
}

;

return_clause: RETURN '(' parameter_type_list ')' ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Return Clause", "Return ( ParameterTypeList )");
   $$ = new C_return_clause($3, localError);
}
| RETURN error_list ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Return Clause", "Return", 
		      "expected ( ParameterTypeList )", true);
   localError->setOriginal();
   $$ = new C_return_clause(0, localError);
}

;

connection_script_definition: CONNECTIONSCRIPT declarator '(' parameter_type_list ')' '{' connection_script_definition_body '}'	{
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Connection Script Definition", "Declarator");
   std::string mes = "ConnectionScript \"";
   mes += $2->getName() + "\"";
   SyntaxError* tdError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   $7->setTdError(tdError);
   $$ = new C_connection_script_definition($2, $4, $7, localError);
}
;

connection_script_definition_body: connection_script_declaration {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Connection Script Definition Body", 
		      "Connection Script Definition");
   $$ = new C_connection_script_definition_body($1, localError);
}
| connection_script_definition_body connection_script_declaration {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Connection Script Definition Body", 
		      "Connection Script Definition Body");
   $$ = new C_connection_script_definition_body($1, $2, localError);
}
;

connection_script_declaration: declaration {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Connection Script Declaration", "Declaration");
   $$ = new C_connection_script_declaration($1, localError);
}
| directive {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Connection Script Declaration", "Directive");
   $$ = new C_connection_script_declaration(false, $1, localError);
}
| RETURN directive {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Connection Script Declaration", "Return Directive");
   $$ = new C_connection_script_declaration(true, $2, localError);
}
;

constant: INT_CONSTANT {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Constant", "Integer");
   $$ = new C_constant($1, localError);
}
| FLOAT_CONSTANT {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Constant", "Float");
   $$ = new C_constant($1, localError);
}
;


query_path_product: query_path DOUBLE_COLON declarator {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "QueryPathProduct", "QueryPath :: Declarator");
   $$ = new C_query_path_product($1, $3, localError);
}
| DOUBLE_COLON declarator {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "QueryPathProduct", ":: Declarator");
   SyntaxError* localError2 = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "QueryPathProduct Auto Generated", "Empty");
   $$ = new C_query_path_product(new C_query_path(localError2), $2, 
				 localError);
}
;
trigger:  query_path_product {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Trigger", "QueryPathProduct");
   $$ = new C_trigger($1, localError);
}
| declarator {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Trigger", "Declarator");
   $$ = new C_trigger($1, localError);
}
| '(' trigger  ')' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @2.first_line, 
		      "Trigger", "( Trigger )");
   $$ = new C_trigger($2, C_trigger::_SINGLE, localError);      
}
| trigger EXP_AND trigger {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Trigger", "Trigger && Trigger");
   $$ = new C_trigger($1, $3, C_trigger::_AND, localError);
}
| trigger EXP_OR trigger {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Trigger", "Trigger || Trigger");
   $$ = new C_trigger($1, $3, C_trigger::_OR, localError);
}
| trigger EXP_XOR trigger {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Trigger", "Trigger ^^ Trigger");
   $$ = new C_trigger($1, $3, C_trigger::_XOR, localError);
}
;

trigger_specifier:  declarator DOT declarator '(' ')' ON trigger ';' {
   std::string mes =  "TriggerSpecifier \"";
   mes += $1->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_trigger_specifier($1, $3, 0, $7, localError);
}
| declarator DOT declarator '(' ndpair_clause_list ')' ON trigger ';' {
   std::string mes =  "TriggerSpecifier \"";
   mes += $1->getName() + "\"";
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, mes.c_str());
   localError->setOriginal();
   $$ = new C_trigger_specifier($1, $3, $5, $8, localError);
}
| PAUSE ON trigger ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "TriggerSpecifier", "Pause on Trigger");
   localError->setOriginal();
   $$ = new C_trigger_specifier("pause", $3, localError);
}
| STOP ON trigger ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "TriggerSpecifier", "Stop on Trigger");
   localError->setOriginal();
   $$ = new C_trigger_specifier("stop", $3, localError);
}
;

service:  query_path_product {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Service", "QueryPathProduct");
   $$ = new C_service($1, localError);
}
| declarator ',' string_literal_list  {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Service", "( declarator , string literal list )");
   $$ = new C_service($1, $3, localError);
}
;

query_path: query_list {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "QueryPath", "QueryList");
   $$ = new C_query_path($1, localError);
}
| repname query_list {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "QueryPath", "RepNameQueryList");
   $$ = new C_query_path($1, $2, localError);
}
;

query: query_field_entry {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Query", "QueryFieldEntry");
   $$ = new C_query($1, localError);
}
| query_field_set {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Query", "QueryFieldSet");
   $$ = new C_query($1, localError);
}
;

query_list: COLON query {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "QueryList", ": Query");
   $$ = new C_query_list($2, localError);
}
| query_list COLON query {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "QueryList", "QueryList : Query");
   $$ = new C_query_list($1, $3, localError);
}
;

query_field_set: '[' query_field_entry_list ']' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @2.first_line, 
		      "QueryFieldSet", "[ QueryFieldEntryList ]");
   $$ = new C_query_field_set($2, localError);
}
;

query_field_entry_list: query_field_entry {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "QueryFieldEntryList", "QueryFieldEntry");
   $$ = new C_query_field_entry_list($1, localError);
}
| query_field_entry_list '|' query_field_entry {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "QueryFieldEntryList", 
		      "QueryFieldEntryList | QueryFieldEntry");
   $$ = new C_query_field_entry_list($1, $3, localError);
}
;

query_field_entry:  '(' ndpair_clause ')'  {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "QueryFieldEntry", "NDPairClause");
   $$ = new C_query_field_entry($2, localError);
}
| string_literal_list {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "QueryFieldEntry", "string literal list");
   $$ = new C_query_field_entry($1, localError);
}
| constant {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "QueryFieldEntry", "constant");
   $$ = new C_query_field_entry($1, localError);
}
| declarator {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "QueryFieldEntry", "Declarator");
   $$ = new C_query_field_entry($1, localError);
}
;

system_call: SYSTEM '(' string_literal_list ')' ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "SystemCall", "system ( \" string literal list \" )");
   localError->setOriginal();
   $$ = new C_system_call($3, localError);
}
| SYSTEM error_list ';' {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "SystemCall", "system", "expected ( \" string \" )", 
		      true);
   localError->setOriginal();
   $$ = new C_system_call(0, localError);
}
;

phase: IDENTIFIER {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "Phase", "Identifier");
   localError->setOriginal();
   $$ = new C_phase(*$1, localError);
   delete $1;
}
;

phase_list: phase {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "PhaseList", "Phase");
   $$ = new C_phase_list($1, localError);
}
| phase_list ',' phase {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "PhaseList", "PhaseList Phase");
   $$ = new C_phase_list($1, $3, localError);
}
;

phase_mapping: IDENTIFIER ARROW IDENTIFIER {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "PhaseMapping", "Identifier -> Identifier");
   localError->setOriginal();
   $$ = new C_phase_mapping(*$1, *$3, localError);
   delete $1;
   delete $3;
}
;

phase_mapping_list: phase_mapping {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "PhaseList", "Phase");
   $$ = new C_phase_mapping_list($1, localError);
}
| phase_mapping_list ',' phase_mapping {
   SyntaxError* localError = 
      new SyntaxError(CURRENTFILE, @1.first_line, 
		      "PhaseMappingList", "PhaseMappingList PhaseMapping");
   $$ = new C_phase_mapping_list($1, $3, localError);
}
;

%%

inline int lenslex(YYSTYPE *lvalp, YYLTYPE *locp, void *context)
{
   return ((LensContext *) context)->
      lexer->lex(lvalp, locp, (LensContext *) context);
}

int main(int argc, char** argv)
{
   /*
   fp_trap(FP_TRAP_SYNC);
   fp_enable(TRP_DIV_BY_ZERO| TRP_OVERFLOW);
   */

  int rank=0;
#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
  if (rank==0) {
    std::cout << ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .\n"
          << ".                                                           .\n"
          << ".  Licensed Materials - Property of IBM                     .\n"
          << ".                                                           .\n"
          << ".  \"Restricted Materials of IBM\"                            .\n"
          << ".                                                           .\n"
          << ".  BMC-YKT-08-23-2011-2                                     .\n"
          << ".                                                           .\n"
          << ".  (C) Copyright IBM Corp. 2005-2014  All rights reserved   .\n"
          << ".                                                           .\n"
          << ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .\n";
  }

   SimInitializer si;
   if (si.execute(&argc, &argv)) {
      return 0;
   } else {
      return 1;
   }
}

void lenserror(char *s)
{
   fprintf(stderr,"%s\n",s);
}

