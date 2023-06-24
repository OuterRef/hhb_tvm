#include "cmd_parse.h"
#include <stdio.h>
int main(int argc, char *argv[])
{
struct cmdline_options *option = cmdline_parser(argc, argv);
printf("index = %d\n", option->rest_line_index);
return 0;
}

