#include "cmdline/cmdline.h"
#include "cmd_parse.h"
#include <iostream>
using namespace std;

static void parse_input_shape(string str, struct cmdline_options *option)
{
  char *input_shape = (char *)malloc(str.length() + 1);
  strcpy(input_shape, str.c_str());
  option->input_number = 0;
  char *ip = strtok(input_shape, ";");
  char shape_buf[CMD_MAX_INPUT][128];
  while (ip != NULL) {
    strcpy(shape_buf[option->input_number], ip);
    option->input_number++;
    ip = strtok(NULL, ";");
  }

  for (int i = 0; i < option->input_number; i++) {
    int dim_count = 0;
    char *sp = strtok(shape_buf[i], ",");
    while (sp != NULL) {
      option->input_shape[i].dims[dim_count] = atoi(sp);
      dim_count++;
      sp = strtok(NULL, ",");
    }
    option->input_shape[i].dim_count = dim_count;
  }
}

struct cmdline_options *cmdline_parser(int argc, char *argv[])
{
  struct cmdline_options *options = (struct cmdline_options *)calloc(sizeof(struct cmdline_options), 1);
  cmdline::parser a;
  a.add<string>("input-shape", 'i', "input shape", false, "");
  a.add("help", 0, "print this message");
  a.footer("bm input_file ...");
  a.set_program_name("hhb_runtime");

  bool ok=a.parse(argc, argv);

  if (argc==1 || a.exist("help")){
    cerr<<a.usage();
    return NULL;
  }
  
  if (!ok){
    cerr<<a.error()<<endl<<a.usage();
    return NULL;
  }

  parse_input_shape(a.get<string>("input-shape"), options);

  options->rest_line_index = argc - a.rest().size();

  return options;
}
