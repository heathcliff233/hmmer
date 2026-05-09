/* hmmnucdb: create a nucleotide GPU database for nhmmer --gpu.
 */
#include <p7_config.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#include "easel.h"
#include "esl_alphabet.h"
#include "esl_getopts.h"
#include "esl_sq.h"
#include "esl_sqio.h"

#include "hmmer.h"
#include "p7_nucdb.h"

static ESL_OPTIONS options[] = {
  /* name           type       default env  range toggles reqs incomp help                                            docgroup */
  { "-h",           eslARG_NONE,  FALSE, NULL, NULL, NULL,   NULL, NULL, "show brief help on version and usage",        1 },
  { "--chunk-size", eslARG_INT, "65536", NULL, "n>0", NULL,  NULL, NULL, "chunk size in residues (default 65536)",      2 },
  { "--overlap",    eslARG_INT,     "0", NULL, "n>=0", NULL, NULL, NULL, "overlap between chunks in residues  [0]",  2 },
  { "--fwd-only",   eslARG_NONE,  FALSE, NULL, NULL, NULL,   NULL, NULL, "store forward strand only (no reverse complement)", 2 },
  { "--informat",   eslARG_STRING, NULL, NULL, NULL, NULL,   NULL, NULL, "assert input <seqfile> is in format <s>",     2 },
  {  0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
};

static char usage[]  = "[options] <seqfile> <nucdb>";
static char banner[] = "build a nucleotide sequence database for nhmmer --gpu";

int
main(int argc, char **argv)
{
  ESL_GETOPTS  *go      = esl_getopts_Create(options);
  ESL_ALPHABET *abc     = NULL;
  ESL_SQFILE   *sqfp    = NULL;
  char         *seqfile = NULL;
  char         *nucdb   = NULL;
  char          errbuf[eslERRBUFSIZE];
  int           fmt     = eslSQFILE_UNKNOWN;
  int           status;
  int64_t       chunk_size;
  int64_t       overlap;
  int           both_strands;

  if (esl_opt_ProcessEnvironment(go)         != eslOK)  { printf("Failed to process environment: %s\n", go->errbuf); goto FAILURE; }
  if (esl_opt_ProcessCmdline(go, argc, argv) != eslOK)  { printf("Failed to parse command line: %s\n",  go->errbuf); goto FAILURE; }
  if (esl_opt_VerifyConfig(go)               != eslOK)  { printf("Failed to parse command line: %s\n",  go->errbuf); goto FAILURE; }

  if (esl_opt_GetBoolean(go, "-h") == TRUE) {
    p7_banner(stdout, argv[0], banner);
    esl_usage(stdout, argv[0], usage);
    puts("\nBasic options:");
    esl_opt_DisplayHelp(stdout, go, 1, 2, 80);
    puts("\nOther options:");
    esl_opt_DisplayHelp(stdout, go, 2, 2, 80);
    exit(0);
  }

  if (esl_opt_ArgNumber(go) != 2) { puts("Incorrect number of command line arguments."); goto FAILURE; }
  seqfile = esl_opt_GetArg(go, 1);
  nucdb   = esl_opt_GetArg(go, 2);
  if (!seqfile || !nucdb) goto FAILURE;

  chunk_size   = esl_opt_GetInteger(go, "--chunk-size");
  overlap      = esl_opt_GetInteger(go, "--overlap");
  both_strands = !esl_opt_GetBoolean(go, "--fwd-only");

  if (esl_opt_IsOn(go, "--informat")) {
    fmt = esl_sqio_EncodeFormat(esl_opt_GetString(go, "--informat"));
    if (fmt == eslSQFILE_UNKNOWN) p7_Fail("%s is not a recognized input sequence file format\n", esl_opt_GetString(go, "--informat"));
  }

  abc = esl_alphabet_Create(eslDNA);
  if (abc == NULL) p7_Fail("Failed to create DNA alphabet\n");

  status = esl_sqfile_OpenDigital(abc, seqfile, fmt, NULL, &sqfp);
  if      (status == eslENOTFOUND) p7_Fail("Failed to open sequence file %s for reading\n",          seqfile);
  else if (status == eslEFORMAT)   p7_Fail("Sequence file %s is empty or misformatted\n",            seqfile);
  else if (status == eslEINVAL)    p7_Fail("Can't autodetect format of a stdin or .gz seqfile");
  else if (status != eslOK)        p7_Fail("Unexpected error %d opening sequence file %s\n", status, seqfile);

  printf("Working...    ");
  fflush(stdout);

  status = p7_nucdb_Write(nucdb, abc, sqfp, chunk_size, overlap, both_strands, errbuf);
  if (status != eslOK) p7_Fail("Failed to write nucdb: %s\n", errbuf);

  printf("done.\n");
  printf("Created nucleotide GPU database %s.nucdb for nhmmer --gpu.\n", nucdb);
  printf("  chunk size: %" PRId64 "  overlap: %" PRId64 "  strands: %s\n",
         chunk_size, overlap, both_strands ? "both" : "forward only");

  esl_sqfile_Close(sqfp);
  esl_alphabet_Destroy(abc);
  esl_getopts_Destroy(go);
  return 0;

 FAILURE:
  esl_usage(stdout, argv[0], usage);
  puts("\nTo see more help on available options, do: hmmnucdb -h");
  esl_getopts_Destroy(go);
  exit(1);
}
